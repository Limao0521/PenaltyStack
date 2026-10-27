"""Parser for B-Human .cfg files, a port of Src/Libs/Streaming/SimpleMap.cpp.

B-Human aborts at startup (red eyes) when a .cfg does not parse, so the
overlay is checked with the same grammar before it is applied or built:

    record  = { literal '=' value ';' }
    value   = literal | '{' record '}' | array
    array   = '[' [ element { ',' element } [','] ] ']'
    element = literal | '{' record '}'
"""

from __future__ import annotations

import sys
from pathlib import Path


class CfgError(ValueError):
    pass


_SINGLE = {"=": "equals", ",": "comma", ";": "semicolon", "[": "lBracket", "]": "rBracket",
           "{": "lBrace", "}": "rBrace"}
_LITERAL_END = set("=,;]}")


class _Lexer:
    def __init__(self, text: str):
        self.text, self.i, self.line, self.col = text, 0, 1, 1
        self.next()

    def _char(self) -> str:
        return self.text[self.i] if self.i < len(self.text) else ""

    def _advance(self) -> None:
        if self._char() == "\n":
            self.line, self.col = self.line + 1, 1
        else:
            self.col += 1
        self.i += 1

    def next(self) -> None:
        while True:
            while self._char() and self._char().isspace():
                self._advance()
            self.where = (self.line, self.col)
            c = self._char()
            if not c:
                self.symbol, self.value = "eof", ""
                return
            if c == "/" and self.text[self.i + 1:self.i + 2] in ("/", "*"):
                if self.text[self.i + 1] == "/":
                    while self._char() and self._char() != "\n":
                        self._advance()
                else:
                    end = self.text.find("*/", self.i + 2)
                    if end < 0:
                        raise CfgError("Unexpected EOF in comment")
                    while self.i < end + 2:
                        self._advance()
                continue
            if c in _SINGLE:
                self.symbol, self.value = _SINGLE[c], c
                self._advance()
                return
            if c == '"':
                start = self.i
                self._advance()
                while self._char() and self._char() != '"':
                    if self._char() == "\\":
                        self._advance()
                    self._advance()
                if not self._char():
                    raise CfgError("Unexpected EOF in string")
                self._advance()
                self.symbol, self.value = "literal", self.text[start:self.i]
                return
            start = self.i
            while self._char() and self._char() not in _LITERAL_END:
                self._advance()
            self.symbol, self.value = "literal", self.text[start:self.i].rstrip()
            return


def _unexpected(lex: _Lexer):
    what = f"literal '{lex.value}'" if lex.symbol == "literal" else f"symbol '{lex.symbol}'"
    raise CfgError(f"({lex.where[0]}, {lex.where[1]}): Unexpected {what}")


def _expect(lex: _Lexer, symbol: str) -> None:
    if lex.symbol != symbol:
        _unexpected(lex)
    lex.next()


def _record(lex: _Lexer) -> dict:
    record = {}
    while lex.symbol == "literal":
        key = lex.value
        lex.next()
        if key in record:
            raise CfgError(f"({lex.where[0]}, {lex.where[1]}): duplicate attribute '{key}'")
        _expect(lex, "equals")
        record[key] = _value(lex)
        _expect(lex, "semicolon")
    return record


def _value(lex: _Lexer):
    if lex.symbol == "literal":
        value = lex.value
        lex.next()
        return value
    if lex.symbol == "lBrace":
        lex.next()
        value = _record(lex)
        _expect(lex, "rBrace")
        return value
    if lex.symbol == "lBracket":
        lex.next()
        items = []
        while lex.symbol in ("literal", "lBrace"):
            if lex.symbol == "literal":
                items.append(lex.value)
                lex.next()
            else:
                lex.next()
                items.append(_record(lex))
                _expect(lex, "rBrace")
            if lex.symbol != "rBracket":
                _expect(lex, "comma")
        _expect(lex, "rBracket")
        return items
    _unexpected(lex)


def parse(text: str) -> dict:
    lex = _Lexer(text)
    record = _record(lex)
    if lex.symbol != "eof":
        _unexpected(lex)
    return record


def check_tree(root: Path) -> list[str]:
    """Return 'path: error' for every .cfg under root that B-Human would reject."""
    errors = []
    for path in sorted(root.rglob("*.cfg")):
        try:
            parse(path.read_text(encoding="utf-8"))
        except CfgError as error:
            errors.append(f"{path.relative_to(root)}{error}")
    return errors


if __name__ == "__main__":
    problems = [e for arg in sys.argv[1:] for e in check_tree(Path(arg))]
    print("\n".join(problems) or "all .cfg files parse")
    raise SystemExit(1 if problems else 0)
