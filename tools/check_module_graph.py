"""Offline check of a B-Human scenario before it reaches a robot.

The NAO only validates its module graph at startup (ModuleGraphCreator::calcShared);
a missing provider means `bhuman` refuses to run on the field. This script repeats
those rules statically for robot/Config/Scenarios/<scenario>/threads.cfg:

  * a provided representation must not also be a default representation;
  * every REQUIRES of every instantiated module (SkillBehaviorControl including
    all of its skills) is a default representation (or alias), provided in the same
    thread, provided by exactly one other thread, or an Upper*/Lower*/... alias;
  * every provider listed in threads.cfg exists and provides that representation;
  * every module/skill with LOADS_PARAMETERS finds its .cfg in the search path of the
    scenario and location (Robots/Default, Locations, Scenarios, Config), and the
    configuration files written by PenaltyStack contain every declared parameter.

    python tools/check_module_graph.py --scenario CopaNAOPenalty --location CopaNAO_CCM2026
"""
import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / 'robot'
SRC = ROBOT / 'Src'
CONFIG = ROBOT / 'Config'
# Configuration files written by PenaltyStack: all parameters must be present.
STRICT_PARAMETERS = ('CopaPenaltyController', 'CopaPenaltyTactics', 'CopaOrangeBallPerceptor')


def strip_comments(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    return re.sub(r'//[^\n]*', '', text)


def balanced(text, start):
    """Text of the parenthesized block that opens at text[start] == '('."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == '(':
            depth += 1
        elif text[index] == ')':
            depth -= 1
            if depth == 0:
                return text[start + 1:index]
    raise ValueError('unbalanced parentheses')


def parameter_names(block):
    match = re.search(r'LOADS_PARAMETERS\(', block)
    if not match:
        return None
    body = balanced(block, match.end() - 1)
    return re.findall(r'\)\s*(\w+)\s*,', body)


def parse_block(block):
    info = {'requires': [], 'provides': [], 'parameters': parameter_names(block)}
    # USES is not part of the module info (_MODULE_INFO_USES is empty): it never needs a provider.
    for kind, name in re.findall(r'\b(REQUIRES|PROVIDES|PROVIDES_WITHOUT_MODIFY|SELECTS)\((\w+)\)', block):
        if kind == 'REQUIRES':
            info['requires'].append(name)
        elif kind == 'SELECTS':
            info['requires'] += ['Lower' + name, 'Upper' + name]
            info['provides'].append(name)
        else:
            info['provides'].append(name)
    return info


def collect_modules():
    modules, skills = {}, {}
    for path in list(SRC.joinpath('Modules').rglob('*.h')) + list(SRC.joinpath('Modules').rglob('*.cpp')):
        text = strip_comments(path.read_text(encoding='utf-8', errors='replace'))
        for macro, target in (('MODULE', modules), ('SKILL_IMPLEMENTATION', skills)):
            for match in re.finditer(r'(?<![\w#])%s\((\w+)\s*,' % macro, text):
                name = match.group(1)
                if name in ('Representation', 'Name'):
                    continue
                target[name] = parse_block(balanced(text, match.end() - len(name) - 2))
        for match in re.finditer(r'(?<![\w#])ALIAS(?:_MODULE)?\((\w+)\)\s*;', text):
            rep = match.group(1)
            modules['Perception%sProvider' % rep] = {'requires': ['Lower' + rep, 'Upper' + rep],
                                                     'provides': [rep], 'parameters': None}
    # SkillBehaviorControl::getExtModuleInfo: the union with the requirements of all skills.
    behavior = modules['SkillBehaviorControl']
    for skill in skills.values():
        behavior['requires'] += [r for r in skill['requires'] if r not in behavior['requires']]
    return modules, skills


def parse_threads(path):
    text = strip_comments(path.read_text(encoding='utf-8'))
    match = re.search(r'defaultRepresentations\s*=\s*\[(.*?)\]', text, re.S)
    defaults = re.findall(r'\w+', match.group(1)) if match else []
    threads = []
    for block in re.finditer(r'name\s*=\s*(\w+);.*?representationProviders\s*=\s*\[(.*?)\];', text, re.S):
        providers = re.findall(r'\{\s*representation\s*=\s*(\w+);\s*provider\s*=\s*(\w+);\s*\}', block.group(2))
        threads.append({'name': block.group(1), 'providers': providers})
    return defaults, threads


def search_path(scenario, location):
    dirs = []
    if location != 'Default':
        dirs.append(CONFIG / 'Locations' / location)
    if scenario != 'Default':
        dirs.append(CONFIG / 'Scenarios' / scenario)
    dirs += [CONFIG / 'Robots/Default', CONFIG / 'Locations/Default', CONFIG / 'Scenarios/Default', CONFIG]
    return dirs


def find_cfg(name, dirs, prefix=''):
    file = prefix + name[0].lower() + name[1:] + '.cfg'
    for directory in dirs:
        if (directory / file).is_file():
            return directory / file
    return None


def check(scenario, location):
    errors = []
    modules, skills = collect_modules()
    threads_file = CONFIG / 'Scenarios' / scenario / 'threads.cfg'
    if not threads_file.is_file():
        threads_file = CONFIG / 'Scenarios/Default/threads.cfg'
    defaults, threads = parse_threads(threads_file)
    names = [t['name'] for t in threads]
    used = set()
    for index, thread in enumerate(threads):
        for representation, provider in thread['providers']:
            if representation in defaults:
                errors.append('%s: %s is also provided by default' % (thread['name'], representation))
            module = modules.get(provider)
            if module is None:
                errors.append('%s: provider %s of %s does not exist' % (thread['name'], provider, representation))
                continue
            if representation not in module['provides']:
                errors.append('%s: %s does not provide %s' % (thread['name'], provider, representation))
            used.add(provider)
            for requirement in module['requires']:
                if requirement in defaults or any(requirement == n + d for n in names for d in defaults):
                    continue
                if any(rep == requirement for rep, _ in thread['providers']):
                    continue
                others = [t for i, t in enumerate(threads) if i != index and any(rep == requirement for rep, _ in t['providers'])]
                alias = next((t for i, t in enumerate(threads) if i != index and requirement.startswith(t['name'])
                              and len(t['name']) < len(requirement)), None)
                alias_ok = alias is not None and any(alias['name'] + rep == requirement for rep, _ in alias['providers'])
                if len(others) > 1:
                    errors.append('%s: %s is provided by multiple threads' % (thread['name'], requirement))
                elif not others and not alias_ok:
                    errors.append('%s: no provider for %s required by %s' % (thread['name'], requirement, provider))
    dirs = search_path(scenario, location)
    # Skill parameters are read from BehaviorControl/<skill>.cfg (SkillRegistryBase), modules' from <module>.cfg.
    configured = [(name, modules[name], '') for name in sorted(used)]
    if 'SkillBehaviorControl' in used:
        configured += [(name, info, 'BehaviorControl/') for name, info in sorted(skills.items())]
        if find_cfg('Skills', dirs, 'BehaviorControl/') is None:
            errors.append('missing BehaviorControl/skills.cfg')
    for name, info, prefix in configured:
        if info['parameters'] is None:
            continue
        cfg = find_cfg(name, dirs, prefix)
        if cfg is None:
            errors.append('missing configuration file for %s (%s.cfg)' % (name, name[0].lower() + name[1:]))
        elif name in STRICT_PARAMETERS:
            keys = set(re.findall(r'^\s*(\w+)\s*=', strip_comments(cfg.read_text(encoding='utf-8')), re.M))
            for parameter in info['parameters']:
                if parameter not in keys:
                    errors.append('%s: parameter %s missing in %s' % (name, parameter, cfg.relative_to(ROBOT)))
    return errors, sorted(used)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Static check of a B-Human scenario module graph.')
    parser.add_argument('--scenario', default='CopaNAOPenalty')
    parser.add_argument('--location', default='CopaNAO_CCM2026')
    args = parser.parse_args(argv)
    errors, used = check(args.scenario, args.location)
    for error in errors:
        print('ERROR ' + error)
    print('%s/%s: %d modules, %d errors' % (args.scenario, args.location, len(used), len(errors)))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
