''' core/input_reader.py '''
import sys
import numpy as np
from enum import Enum, auto
from .data_types import NODE_DTYPE, FACE_DTYPE, CELL_DTYPE, CONSTRAINT_DTYPE, SOURCE_DTYPE

class TokenType(Enum):
    EOF = auto(); NUMBER = auto(); IDENTIFIER = auto()
    LBRACKET = auto(); RBRACKET = auto()

class Token:
    def __init__(self, text, kind): self.text = text; self.kind = kind

class Lexer:
    def __init__(self, input_text):
        self.source = input_text + '\n'
        self.curPos = -1
        self.curChar = ''
        self.nextChar()
    
    def nextChar(self):
        self.curPos += 1
        if self.curPos >= len(self.source): self.curChar = '\0'
        else: self.curChar = self.source[self.curPos]

    def peek(self):
        if self.curPos + 1 >= len(self.source): return '\0'
        return self.source[self.curPos + 1]

    def abort(self, msg): sys.exit("Lexing error: " + msg)
    def skipWhitespace(self):
        while self.curChar in [' ', '\t', '\n', '\r']: self.nextChar()
    def skipComment(self):
        if self.curChar == '#':
            while self.curChar != '\n' and self.curChar != '\0': self.nextChar()

    def getToken(self):
        self.skipWhitespace()
        while self.curChar == '#': self.skipComment(); self.skipWhitespace()

        if self.curChar == '\0': return Token('', TokenType.EOF)
        elif self.curChar == '[': self.nextChar(); return Token('[', TokenType.LBRACKET)
        elif self.curChar == ']': self.nextChar(); return Token(']', TokenType.RBRACKET)
        elif self.curChar.isdigit() or self.curChar == '.' or self.curChar == '-':
            start = self.curPos
            while self.peek().isdigit() or self.peek() == '.' or self.peek() == '-': self.nextChar()
            text = self.source[start : self.curPos+1]; self.nextChar(); return Token(text, TokenType.NUMBER)
        elif self.curChar.isalpha():
            start = self.curPos
            while self.peek().isalnum() or self.peek() == '_': self.nextChar()
            text = self.source[start : self.curPos+1]; self.nextChar(); return Token(text, TokenType.IDENTIFIER)
        else: self.abort(f"Unknown token: {self.curChar}")

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.curToken = None; self.peekToken = None
        self.nextToken(); self.nextToken()
        
        self.nodes = []; self.faces = []; self.cells = []; self.constraints = []; self.sources = []
        self.constraint_types = {'fixed':0, 'line':1, 'circle':2}
        self.source_types = {'global':0, 'box':1}

    def checkToken(self, kind): return self.kind == kind
    def nextToken(self):
        self.curToken = self.peekToken; self.peekToken = self.lexer.getToken()
        self.kind = self.curToken.kind if self.curToken else None
    def match(self, kind):
        if not self.checkToken(kind): 
            got = self.curToken.text if self.curToken else 'None'
            self.abort(f"Expected {kind}, got {got}")
        self.nextToken()
    def abort(self, msg): sys.exit(msg)

    def program(self):
        while not self.checkToken(TokenType.EOF): self.section()

    def section(self):
        self.match(TokenType.LBRACKET)
        name = self.curToken.text.lower()
        self.match(TokenType.IDENTIFIER)
        self.match(TokenType.RBRACKET)

        while self.checkToken(TokenType.NUMBER):
            if name == "nodes":
                nid = int(self.curToken.text); self.match(TokenType.NUMBER)
                x = float(self.curToken.text); self.match(TokenType.NUMBER)
                y = float(self.curToken.text); self.match(TokenType.NUMBER)
                self.nodes.append((nid, x, y))

            elif name == "constraints":
                cid = int(self.curToken.text); self.match(TokenType.NUMBER)
                ctype = 1
                if self.checkToken(TokenType.IDENTIFIER):
                    ctype = self.constraint_types.get(self.curToken.text.lower(), 1)
                    self.match(TokenType.IDENTIFIER)
                elif self.checkToken(TokenType.NUMBER):
                    ctype = int(self.curToken.text)
                    self.match(TokenType.NUMBER)
                p1 = float(self.curToken.text); self.match(TokenType.NUMBER)
                p2 = float(self.curToken.text); self.match(TokenType.NUMBER)
                p3 = float(self.curToken.text); self.match(TokenType.NUMBER)
                self.constraints.append((cid, ctype, 0, p1, p2, p3))

            elif name == "sources":
                sid = int(self.curToken.text); self.match(TokenType.NUMBER)
                stype = 0
                if self.checkToken(TokenType.IDENTIFIER):
                    stype = self.source_types.get(self.curToken.text.lower(), 0)
                    self.match(TokenType.IDENTIFIER)
                x1 = float(self.curToken.text); self.match(TokenType.NUMBER)
                y1 = float(self.curToken.text); self.match(TokenType.NUMBER)
                x2 = float(self.curToken.text); self.match(TokenType.NUMBER)
                y2 = float(self.curToken.text); self.match(TokenType.NUMBER)
                h  = float(self.curToken.text); self.match(TokenType.NUMBER)
                self.sources.append((sid, stype, x1, y1, x2, y2, h))

            elif name == "faces":
                fid = int(self.curToken.text); self.match(TokenType.NUMBER)
                n1 = int(self.curToken.text); self.match(TokenType.NUMBER)
                n2 = int(self.curToken.text); self.match(TokenType.NUMBER)
                tag = int(self.curToken.text); self.match(TokenType.NUMBER)
                # Parse Segments (Optional, defaults to 10 if missing, but strictly enforced by parser logic)
                # To be safe with your format, we assume it's always there if we agreed on the format.
                segs = int(self.curToken.text); self.match(TokenType.NUMBER)
                self.faces.append((fid, n1, n2, tag, segs))

            elif name == "cells":
                self.nextToken() # Skip
            else: self.abort(f"Unknown section: {name}")

    def get_arrays(self):
        return {
            "nodes": np.array(self.nodes, dtype=NODE_DTYPE),
            "faces": np.array(self.faces, dtype=FACE_DTYPE),
            "constraints": np.array(self.constraints, dtype=CONSTRAINT_DTYPE),
            "sources": np.array(self.sources, dtype=SOURCE_DTYPE)
        }

def input_reader(filename):
    with open(filename, 'r') as f: 
        parser = Parser(Lexer(f.read()))
        
        # CRITICAL FIX: Run the parsing loop!
        parser.program()  
        
        return parser.get_arrays()
    
    
    
    