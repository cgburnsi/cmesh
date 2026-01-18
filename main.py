import sys
import numpy as np
from enum import Enum, auto

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
        self.skipWhitespace()
        self.skipComment()
        self.skipWhitespace() # Skip whitespace again in case comment ended with newline

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
            # Parsing Identifiers (nodes, edges, etc.)
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
        self.edges_list = []
        self.cells_list = []

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

        print(f"Parsing section: {section_name}")

        # Parse data based on which section we are in
        while self.checkToken(TokenType.NUMBER):
            if section_name == "nodes":
                # Expecting: ID, X, Y
                nid = float(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                x = float(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                y = float(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                self.nodes_list.append([nid, x, y])

            elif section_name == "edges":
                # Expecting: ID, N1, N2
                eid = int(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                n1 = int(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                n2 = int(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                self.edges_list.append([eid, n1, n2])

            elif section_name == "cells":
                # Expecting: ID, E1, E2, E3
                cid = int(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                e1 = int(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                e2 = int(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                e3 = int(self.curToken.text)
                self.match(TokenType.NUMBER)
                
                self.cells_list.append([cid, e1, e2, e3])
            
            else:
                self.abort(f"Unknown section header: {section_name}")

    # Helper to return final arrays
    def get_arrays(self):
        return {
            "nodes": np.array(self.nodes_list),
            "edges": np.array(self.edges_list),
            "cells": np.array(self.cells_list)
        }

if __name__ == '__main__':


    with open('geom1.inp', 'r') as input_file:
        source = input_file.read()    
        
    # Initialize Lexer
    lexer = Lexer(source)
    
    # Initialize Parser
    parser = Parser(lexer)
    
    # Run Parser
    parser.program()
    
    # Retrieve Data
    data = parser.get_arrays()
    
    print("\n--- Output Arrays ---")
    print("Nodes:\n", data["nodes"])
    print("Edges:\n", data["edges"])
    print("Cells:\n", data["cells"])





