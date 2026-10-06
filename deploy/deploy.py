"""Entrega Ethernet a directorio aislado; por defecto solo muestra pasos."""
import argparse
import hashlib
import pathlib
import re
import subprocess


def deploy(archive_path, host, apply=False):
    archive = pathlib.Path(archive_path).resolve()
    if not archive.is_file():
        raise ValueError('archive does not exist')
    if not re.match(r'^[A-Za-z0-9][A-Za-z0-9.:-]*$', host):
        raise ValueError('invalid host')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    release = digest[:16]
    remote = 'nao@' + host
    base = '/home/nao/PenaltyStack'
    commands = [
        (['ssh', remote, 'mkdir -p ' + base + '/releases/' + release], False),
        (['ssh', remote, 'cat > ' + base + '/releases/' + release + '/package.tar.gz'], True),
        (['ssh', remote, 'cd ' + base + '/releases/' + release +
         ' && test "$(sha256sum package.tar.gz | cut -d\' \' -f1)" = ' + digest +
         ' && tar -xzf package.tar.gz && ln -sfn releases/' + release + ' ' + base + '/current'], False),
    ]
    for command, upload in commands:
        print(' '.join(command) + (' < ' + str(archive) if upload else ''))
        if apply:
            if upload:
                with archive.open('rb') as content:
                    subprocess.check_call(command, stdin=content)
            else:
                subprocess.check_call(command)
    return digest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--apply', action='store_true', help='copy to the NAO by Ethernet')
    args = parser.parse_args()
    print('sha256=' + deploy(args.archive, args.host, args.apply))
