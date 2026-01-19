''' core/input_reader.py '''
import sys
import numpy as np
from enum import Enum, auto
from core import NODE_DTYPE, FACE_DTYPE, CELL_DTYPE, CONSTRAINT_DTYPE

class TokenType(Enum):
    EOF = auto()
    NUMBER = auto()
    IDENTIFIER = auto()
    LBRACKET = auto()
    RBRACKET = auto()

class Token:
    def __init__(self, token_text, token_kind):
        self.text = token_text
        self.kind = token_kind
    def __repr__(self):
        return f"{self.kind.name}: {self.text}"

class Lexer:
    def __init__(self, input_text):
        self.source = input_text + '\n'
        self.curPos = -1
        self.curChar = ''
        self.nextChar()

    def nextChar(self):
        self.curPos += 1
        if self.curPos >= len(self.source):
            self.curChar = '\0'
        else:
            self.curChar = self.source[self.curPos]

    def peek(self):
        if self.curPos + 1 >= len(self.source):
            return '\0'
        return self.source[self.curPos + 1]

    def abort(self, message):
        sys.exit("Lexing error: " + message)

    def skipWhitespace(self):
        while self.curChar in [' ', '\t', '\n', '\r']:
            self.nextChar()

    def skipComment(self):
        if self.curChar == '#':
            while self.curChar != '\n' and self.curChar != '\0':
                self.nextChar()

    def getToken(self):
        self.skipWhitespace()
        while self.curChar == '#':
            self.skipComment()
            self.skipWhitespace()

        if self.curChar == '\0':
            return Token('', TokenType.EOF)
        elif self.curChar == '[':
            self.nextChar()
            return Token('[', TokenType.LBRACKET)
        elif self.curChar == ']':
            self.nextChar()
            return Token(']', TokenType.RBRACKET)
            
        # --- FIX START: Numbers ---
        elif self.curChar.isdigit() or self.curChar == '.' or self.curChar == '-':
            startPos = self.curPos
            while self.peek().isdigit() or self.peek() == '.' or self.peek() == '-':
                self.nextChar()
            tokenText = self.source[startPos : self.curPos + 1]
            # CRITICAL FIX: Advance past the last digit
            self.nextChar() 
            return Token(tokenText, TokenType.NUMBER)
            
        # --- FIX START: Identifiers ---
        elif self.curChar.isalpha():
            startPos = self.curPos
            while self.peek().isalnum() or self.peek() == '_':
                self.nextChar()
            tokenText = self.source[startPos : self.curPos + 1]
            # CRITICAL FIX: Advance past the last letter
            self.nextChar() 
            return Token(tokenText, TokenType.IDENTIFIER)
        else:
            self.abort("Unknown token: " + self.curChar)

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.curToken = None
        self.peekToken = None
        self.nextToken()
        self.nextToken()

        self.nodes_list = []
        self.faces_list = []
        self.cells_list = []
        self.constraints_list = []
        
        self.constraint_types = {'fixed': 0, 'line': 1, 'circle': 2}

    def checkToken(self, kind): return self.kind == kind
    def nextToken(self):
        self.curToken = self.peekToken
        self.peekToken = self.lexer.getToken()
        self.kind = self.curToken.kind if self.curToken else None
    def match(self, kind):
        if not self.checkToken(kind): 
            # Improved error message to see WHAT we actually got
            got_text = self.curToken.text if self.curToken else "None"
            self.abort(f"Expected {kind}, got {self.curToken.kind} ('{got_text}')")
        self.nextToken()
    def abort(self, msg): sys.exit(msg)

    def program(self):
        while not self.checkToken(TokenType.EOF):
            self.section()

    def section(self):
        self.match(TokenType.LBRACKET)
        section_name = self.curToken.text.lower()
        self.match(TokenType.IDENTIFIER)
        self.match(TokenType.RBRACKET)

        while self.checkToken(TokenType.NUMBER):
            if section_name == "nodes":
                nid = int(self.curToken.text); self.match(TokenType.NUMBER)
                x = float(self.curToken.text); self.match(TokenType.NUMBER)
                y = float(self.curToken.text); self.match(TokenType.NUMBER)
                self.nodes_list.append((nid, x, y))

            elif section_name == "constraints":
                cid = int(self.curToken.text); self.match(TokenType.NUMBER)
                
                ctype = 1
                if self.checkToken(TokenType.IDENTIFIER):
                    t_str = self.curToken.text.lower()
                    ctype = self.constraint_types.get(t_str, 1)
                    self.match(TokenType.IDENTIFIER)
                elif self.checkToken(TokenType.NUMBER):
                    ctype = int(self.curToken.text)
                    self.match(TokenType.NUMBER)

                p1 = float(self.curToken.text); self.match(TokenType.NUMBER)
                p2 = float(self.curToken.text); self.match(TokenType.NUMBER)
                p3 = float(self.curToken.text); self.match(TokenType.NUMBER)
                self.constraints_list.append((cid, ctype, 0, p1, p2, p3))

            elif section_name == "faces":
                fid = int(self.curToken.text); self.match(TokenType.NUMBER)
                n1 = int(self.curToken.text); self.match(TokenType.NUMBER)
                n2 = int(self.curToken.text); self.match(TokenType.NUMBER)
                tag = int(self.curToken.text); self.match(TokenType.NUMBER)
                self.faces_list.append((fid, n1, n2, tag))

            elif section_name == "cells":
                cid = int(self.curToken.text); self.match(TokenType.NUMBER)
                e1 = int(self.curToken.text); self.match(TokenType.NUMBER)
                e2 = int(self.curToken.text); self.match(TokenType.NUMBER)
                e3 = int(self.curToken.text); self.match(TokenType.NUMBER)
                self.cells_list.append((cid, e1, e2, e3))
            
            else:
                self.abort(f"Unknown section: {section_name}")

    def get_arrays(self):
        return {
            "nodes": np.array(self.nodes_list, dtype=NODE_DTYPE),
            "faces": np.array(self.faces_list, dtype=FACE_DTYPE),
            "cells": np.array(self.cells_list, dtype=CELL_DTYPE),
            "constraints": np.array(self.constraints_list, dtype=CONSTRAINT_DTYPE)
        }

def input_reader(filename):
    with open(filename, 'r') as f: source = f.read()
    lexer = Lexer(source)
    parser = Parser(lexer)
    parser.program()
    return parser.get_arrays()