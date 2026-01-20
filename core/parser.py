''' core/parser.py
    --------------
    Recursive Descent Parser for the SnapMesh Input Format.
    
    This module consumes the stream of tokens from the Lexer and constructs
    the structured NumPy arrays (Nodes, Faces, Fields) required by the core 
    engine. It uses a "Dispatch Table" pattern to cleanly separate the logic 
    for each section of the input file.

    Attribution & License:
    ----------------------
    The recursive descent architecture used here is adapted from the 
    "Teeny Tiny Compiler" project by Austen Hensley.  I have not used much of
    his emitter code since I am sending data directly to the numpy arrays.
    
    Source: https://github.com/austenhensley
    
    Copyright (c) 2024 Austen Hensley
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
'''
import sys
import numpy as np
from .data_types import NODE_DTYPE, FACE_DTYPE, CELL_DTYPE, CONSTRAINT_DTYPE, FIELD_DTYPE
from .lexer import Lexer, TokenType 

class Parser:
    """
    Parses the input file into structured data arrays.
    
    Architecture:
    - Recursive Descent: The 'program' method calls 'process_section', which 
      dispatches to specific handlers (e.g., 'parse_nodes').
    - Dispatch Table: Section handlers are registered in 'self.section_handlers',
      making it easy to add new features (like Materials or BCs) without 
      breaking the core loop.
    """
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
        
    def abort(self, msg): 
        sys.exit(msg)

    # --- Value Extraction Helpers ---
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
        segs = self.expect_int() 
        self.data['faces'].append((fid, n1, n2, tag, segs))

    def parse_constraints(self):
        cid   = self.expect_int()
        ctype = self.expect_enum(self.constraint_types, 1) 
        p1    = self.expect_float()
        p2    = self.expect_float()
        p3    = self.expect_float()
        self.data['constraints'].append((cid, ctype, 0, p1, p2, p3))

    def parse_fields(self):
        sid   = self.expect_int()
        stype = self.expect_enum(self.field_types, 0)
        x1    = self.expect_float()
        y1    = self.expect_float()
        x2    = self.expect_float()
        y2    = self.expect_float()
        v     = self.expect_float()
        self.data['fields'].append((sid, stype, x1, y1, x2, y2, v))

    def skip_section(self):
        self.nextToken() 

    def get_arrays(self):
        return {
            "nodes":       np.array(self.data['nodes'], dtype=NODE_DTYPE),
            "faces":       np.array(self.data['faces'], dtype=FACE_DTYPE),
            "constraints": np.array(self.data['constraints'], dtype=CONSTRAINT_DTYPE),
            "fields":      np.array(self.data['fields'], dtype=FIELD_DTYPE)
        }

# This is the public facing code for loading a data file
def input_reader(filename):
    with open(filename, 'r') as f: 
        lexer  = Lexer(f.read())
        parser = Parser(lexer)
        parser.program()
        
        return parser.get_arrays()
    
    
    
    
    
    
    