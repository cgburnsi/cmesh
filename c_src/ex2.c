#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

#define MAX_NODES 100
#define MAX_EDGES 100
#define MAX_LINE 256

typedef struct { int id; double x; double y; } Node;
typedef struct { int id; int n1; int n2; } Edge;

// --- GLOBAL STORAGE (For simplicity in this step) ---
Node nodes[MAX_NODES];
Edge edges[MAX_EDGES];
int node_count = 0;
int edge_count = 0;

// --- HELPER 1: TRIM WHITESPACE ---
// Modifies the string in-place to remove leading/trailing spaces
char* trim(char *str) {
    char *end;

    // Trim leading space
    while(isspace((unsigned char)*str)) str++;

    if(*str == 0) return str; // All spaces

    // Trim trailing space
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;

    // Write new null terminator
    *(end+1) = 0;

    return str;
}

// --- HELPER 2: ROBUST LINE READER ---
// Reads until it finds a non-empty, non-comment line (or EOF)
int get_clean_line(FILE *f, char *buffer, int size) {
    while (fgets(buffer, size, f)) {
        // 1. Remove comments
        char *comment_start = strchr(buffer, '#');
        if (comment_start) *comment_start = '\0';

        // 2. Trim whitespace
        char *clean = trim(buffer);

        // 3. If line is empty after cleaning, skip it
        if (strlen(clean) == 0) continue;

        // 4. Move cleaned text to start of buffer (if trim shifted pointer)
        if (clean != buffer) memmove(buffer, clean, strlen(clean) + 1);
        
        return 1; // Found a good line
    }
    return 0; // End of File
}

// --- PARSERS ---
void parse_node(char *line) {
    if (node_count >= MAX_NODES) return;
    
    // We use temporary vars so we don't corrupt memory on failure
    int id; double x, y;
    
    if (sscanf(line, "%d %lf %lf", &id, &x, &y) == 3) {
        nodes[node_count].id = id;
        nodes[node_count].x = x;
        nodes[node_count].y = y;
        node_count++;
    } else {
        printf("!! WARNING: Malformed Node Line: '%s'\n", line);
    }
}

void parse_edge(char *line) {
    if (edge_count >= MAX_EDGES) return;
    
    int id, n1, n2;
    if (sscanf(line, "%d %d %d", &id, &n1, &n2) == 3) {
        edges[edge_count].id = id;
        edges[edge_count].n1 = n1;
        edges[edge_count].n2 = n2;
        edge_count++;
    } else {
        printf("!! WARNING: Malformed Edge Line: '%s'\n", line);
    }
}

// --- MAIN ---
typedef enum { STATE_NONE, STATE_NODES, STATE_EDGES } ParseState;

int main() {
    FILE *file = fopen("ex2.inp", "r");
    if (!file) { perror("Error"); return 1; }

    char line[MAX_LINE];
    ParseState state = STATE_NONE;

    // The logic is now much cleaner
    while (get_clean_line(file, line, sizeof(line))) {
        
        // CHECK FOR HEADER
        if (line[0] == '[') {
            // Check specific headers
            // We use strstr to see if the tag exists inside the brackets
            if (strstr(line, "nodes")) {
                state = STATE_NODES;
                printf("Switched to NODES\n");
            } 
            else if (strstr(line, "edges")) {
                state = STATE_EDGES;
                printf("Switched to EDGES\n");
            } 
            else {
                state = STATE_NONE;
                printf("!! WARNING: Unknown Header '%s'\n", line);
            }
            continue;
        }

        // DISPATCH DATA
        switch (state) {
            case STATE_NODES: parse_node(line); break;
            case STATE_EDGES: parse_edge(line); break;
            default: 
                // Useful for debugging lost data
                printf("Ignoring line outside section: %s\n", line);
                break;
        }
    }

    fclose(file);

    // Verify
    printf("\n--- Final Data ---\n");
    printf("Nodes: %d\n", node_count);
    printf("Edges: %d\n", edge_count);

    return 0;
}
