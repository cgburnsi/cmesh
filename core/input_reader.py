''' core/input_reader.py '''
import sys
import numpy as np
from enum import Enum, auto
from .data_types import NODE_DTYPE, FACE_DTYPE, CELL_DTYPE, CONSTRAINT_DTYPE, FIELD_DTYPE

class TokenType(Enum):
    EOF = auto(); NUMBER = auto(); IDENTIFIER = auto()
    LBRACKET = auto(); RBRACKET = auto()

class Token:
    def __init__(self, text, kind): self.text = text; self.kind = kind

class Lexer:
    # (Lexer remains unchanged - it works perfectly)
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
        
        # 1. Data Storage
        self.data = {
            "nodes": [], 
            "faces": [], 
            "constraints": [], 
            "fields": []
        }
        
        # 2. Lookup Tables (Enums for Input File Strings)
        self.constraint_types = {'fixed':0, 'line':1, 'circle':2}
        self.field_types = {'global':0, 'box':1}

        # 3. Dispatch Table: Maps Section Name -> Handler Method
        self.section_handlers = {
            "nodes": self.parse_nodes,
            "faces": self.parse_faces,
            "constraints": self.parse_constraints,
            "fields": self.parse_fields,
            "cells": self.skip_section # Placeholder for future
        }

    # --- Core Parsing Helpers ---
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

    # --- Value Extraction Helpers (Cleans up the code significantly) ---
    def expect_int(self):
        val = int(self.curToken.text); self.match(TokenType.NUMBER)
        return val
    def expect_float(self):
        val = float(self.curToken.text); self.match(TokenType.NUMBER)
        return val
    def expect_enum(self, mapping_dict, default_val):
        val = default_val
        if self.checkToken(TokenType.IDENTIFIER):
            text = self.curToken.text.lower()
            val = mapping_dict.get(text, default_val)
            self.match(TokenType.IDENTIFIER)
        elif self.checkToken(TokenType.NUMBER):
            val = int(self.curToken.text); self.match(TokenType.NUMBER)
        return val

    # --- Main Loop ---
    def program(self):
        while not self.checkToken(TokenType.EOF): 
            self.process_section()

    def process_section(self):
        self.match(TokenType.LBRACKET)
        section_name = self.curToken.text.lower()
        self.match(TokenType.IDENTIFIER)
        self.match(TokenType.RBRACKET)

        # Dispatcher Logic
        handler = self.section_handlers.get(section_name)
        if not handler:
            self.abort(f"Unknown section: [{section_name}]")
        
        # Run the specific handler until we run out of numbers
        while self.checkToken(TokenType.NUMBER):
            handler()

    # --- Individual Handlers ---
    def parse_nodes(self):
        nid = self.expect_int()
        x   = self.expect_float()
        y   = self.expect_float()
        self.data['nodes'].append((nid, x, y))

    def parse_faces(self):
        fid  = self.expect_int()
        n1   = self.expect_int()
        n2   = self.expect_int()
        tag  = self.expect_int()
        segs = self.expect_int() # Segments is now mandatory in logic
        self.data['faces'].append((fid, n1, n2, tag, segs))

    def parse_constraints(self):
        cid   = self.expect_int()
        ctype = self.expect_enum(self.constraint_types, 1) # Default to line (1)
        p1    = self.expect_float()
        p2    = self.expect_float()
        p3    = self.expect_float()
        # Note: 'target' is currently hardcoded 0, can be updated later
        self.data['constraints'].append((cid, ctype, 0, p1, p2, p3))

    def parse_fields(self):
        sid   = self.expect_int()
        stype = self.expect_enum(self.field_types, 0) # Default to global (0)
        x1    = self.expect_float()
        y1    = self.expect_float()
        x2    = self.expect_float()
        y2    = self.expect_float()
        v     = self.expect_float()
        self.data['fields'].append((sid, stype, x1, y1, x2, y2, v))

    def skip_section(self):
        # Just consume the tokens for a line if needed, or implement generic skipper
        # For now, we just consume the first token to avoid infinite loops if buggy
        self.nextToken() 

    def get_arrays(self):
        return {
            "nodes":       np.array(self.data['nodes'], dtype=NODE_DTYPE),
            "faces":       np.array(self.data['faces'], dtype=FACE_DTYPE),
            "constraints": np.array(self.data['constraints'], dtype=CONSTRAINT_DTYPE),
            "fields":      np.array(self.data['fields'], dtype=FIELD_DTYPE)
        }

def input_reader(filename):
    with open(filename, 'r') as f: 
        parser = Parser(Lexer(f.read()))
        parser.program()
        return parser.get_arrays()