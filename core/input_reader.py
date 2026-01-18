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
        return f'Token Type: {self.kind}, Token Text: {self.text}'

class Lexer:
    def __init__(self, input_text):
        self.source = input_text + '\n' # Append newline to simplify EOF handling
        self.curChar = ''
        self.curPos = -1
        self.nextChar()

    # Process the next character
    def nextChar(self):
        self.curPos += 1
        if self.curPos >= len(self.source):
            self.curChar = '\0'  # EOF
        else:
            self.curChar = self.source[self.curPos]

    # Return the lookahead character
    def peek(self):
        if self.curPos + 1 >= len(self.source):
            return '\0'
        return self.source[self.curPos + 1]

    # Invalid syntax handling
    def abort(self, message):
        sys.exit("Lexing error: " + message)

    # Skip whitespace
    def skipWhitespace(self):
        while self.curChar in [' ', '\t', '\n', '\r']:
            self.nextChar()

    # Skip comments (Modified for your '#' style)
    def skipComment(self):
        if self.curChar == '#':
            while self.curChar != '\n':
                self.nextChar()

    # Return the next token
    def getToken(self):
        while True:
            self.skipWhitespace()
            if self.curChar == '#':
                self.skipComment()
            else:
                break
        token = None

        if self.curChar == '\0':
            token = Token('', TokenType.EOF)
        elif self.curChar == '[':
            token = Token(self.curChar, TokenType.LBRACKET)
            self.nextChar()
        elif self.curChar == ']':
            token = Token(self.curChar, TokenType.RBRACKET)
            self.nextChar()
        elif self.curChar.isdigit() or self.curChar == '.':
            # Parsing Numbers (Integers and Floats)
            startPos = self.curPos
            while self.peek().isdigit() or self.peek() == '.':
                self.nextChar()
            tokenText = self.source[startPos : self.curPos + 1]
            token = Token(tokenText, TokenType.NUMBER)
            self.nextChar()
        elif self.curChar.isalpha():
            # Parsing Identifiers (nodes, faces, etc.)
            startPos = self.curPos
            while self.peek().isalnum():
                self.nextChar()
            tokenText = self.source[startPos : self.curPos + 1]
            token = Token(tokenText, TokenType.IDENTIFIER)
            self.nextChar()
        else:
            self.abort("Unknown token: " + self.curChar)

        return token

# --- 3. The Parser ---
class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.curToken = None
        self.peekToken = None
        self.nextToken()
        self.nextToken() # Call twice to initialize current and peek

        # -- Storage for our Data (The "Emitter" target) --
        self.nodes_list = []
        self.faces_list = []
        self.cells_list = []
        self.constraints_list = []  # <--- NEW STORAGE

    # Return true if the current token matches
    def checkToken(self, kind):
        return self.kind == kind

    # Return true if the next token matches
    def checkPeek(self, kind):
        return self.peekToken.kind == kind

    # Try to match current token. If not, error. Advances the token.
    def match(self, kind):
        if not self.checkToken(kind):
            self.abort(f"Expected {kind.name}, got {self.curToken.kind.name}")
        self.nextToken()

    def nextToken(self):
        self.curToken = self.peekToken
        self.peekToken = self.lexer.getToken()
        self.kind = self.curToken.kind if self.curToken else None

    def abort(self, message):
        sys.exit("Parsing error: " + message)

    # --- Grammar Rules ---

    # program ::= { section }
    def program(self):
        print("PROGRAM START")
        while not self.checkToken(TokenType.EOF):
            self.section()
        print("PROGRAM FINISHED")

    # section ::= "[" IDENTIFIER "]" { NUMBER }
    def section(self):
        self.match(TokenType.LBRACKET)
        section_name = self.curToken.text
        self.match(TokenType.IDENTIFIER)
        self.match(TokenType.RBRACKET)

        while self.checkToken(TokenType.NUMBER):
            if section_name == "nodes":
                # IDs must be int to match 'i4' in NODE_DTYPE
                nid = int(self.curToken.text) 
                self.match(TokenType.NUMBER)
                
                # Coordinates are float ('f8')
                x = float(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                y = float(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                # Append as tuple
                self.nodes_list.append((nid, x, y))

            elif section_name == "faces":
                # All integers for FACE_DTYPE
                eid = int(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                n1 = int(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                n2 = int(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                tag = int(self.curToken.text); self.match(TokenType.NUMBER)
                
                self.faces_list.append((eid, n1, n2, tag))

            elif section_name == "constraints":
                # Expecting: ID, TYPE, TARGET, P1, P2, P3
                cid = int(self.curToken.text); self.match(TokenType.NUMBER)
                typ = int(self.curToken.text); self.match(TokenType.NUMBER)
                tgt = int(self.curToken.text); self.match(TokenType.NUMBER)
                
                p1 = float(self.curToken.text); self.match(TokenType.NUMBER)
                p2 = float(self.curToken.text); self.match(TokenType.NUMBER)
                p3 = float(self.curToken.text); self.match(TokenType.NUMBER)

                self.constraints_list.append((cid, typ, tgt, p1, p2, p3))
                
            elif section_name == "cells":
                # All integers for CELL_DTYPE
                cid = int(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                f1 = int(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                f2 = int(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                f3 = int(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                self.cells_list.append((cid, f1, f2, f3))
            
            else:
                self.abort(f"Unknown section header: {section_name}")

    # Helper to return final arrays
    def get_arrays(self):
        # converting lists of tuples to structured NumPy arrays
        return {
            "nodes": np.array(self.nodes_list, dtype=NODE_DTYPE),
            "faces": np.array(self.faces_list, dtype=FACE_DTYPE),
            "cells": np.array(self.cells_list, dtype=CELL_DTYPE),
            "constraints": np.array(self.constraints_list, dtype=CONSTRAINT_DTYPE)
        }








def input_reader(filename):
    
    with open(filename, 'r') as input_file:
        source = input_file.read()    
    
    lexer  = Lexer(source)
    parser = Parser(lexer)
    parser.program()
    
    return parser.get_arrays()



if __name__ == '__main__':
    pass


