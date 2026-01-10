#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// 'Hard Code' The memory requirements to eliminate the dynamic memory work until later.
#define MAX_NODES 100
#define MAX_EDGES 100
#define MAX_LINE_LENGTH 256

typedef struct {
	int id;
	double x;
	double y;
} Node;

typedef struct {
	int id;
	int node1;
	int node2;
} Edge;

typedef enum {
	STATE_NONE,
	STATE_NODES,
	STATE_EDGES,
} ParseState;





int main() {

	/* Information Storage */
	Node nodes[MAX_NODES];
	Edge edges[MAX_EDGES];

	/* Counters */
	int node_count = 0;
	int edge_count = 0;

	/* Open File */
	FILE *file = fopen("ex1.inp", "r");
	if (file == NULL) {
		perror("ERROR: Error Opening File");
		return 1;
	}

	char line[MAX_LINE_LENGTH];
	ParseState current_state = STATE_NONE;

	/* File Reading Loop */
	while (fgets(line, sizeof(line), file)) {
		// Remove trailing newline character if present (for clean printing/logic)
		line[strcspn(line, "\n")] = 0;

		// skip empty lines
		if (strlen(line) == 0) continue;

		// CHECK SECTION HEADER: Is this a section header?
		if (strcmp(line, "nodes") == 0) {
			current_state = STATE_NODES;
			printf("Found Nodes Header...\n");
			continue; // Skips to the next line
		}
		else if (strcmp(line, "edges") == 0) {
			current_state = STATE_EDGES;
			printf("Found Edges Header...\n");
			continue; // Skips to the next line
		}

		// Parse Each Line
		if (current_state == STATE_NODES) {
			sscanf(line, "%d %lf %lf",
				&nodes[node_count].id,
				&nodes[node_count].x,
				&nodes[node_count].y);
			node_count++;	
		}
		else if (current_state == STATE_EDGES) {
			sscanf(line, "%d %d %d",
				&edges[edge_count].id,
				&edges[edge_count].node1,
				&edges[edge_count].node2);
			edge_count++;
		}
	}


	// Output File Summary
	printf("\n--- Summary ---\n");
	printf("Loaded %d nodes:\n", node_count);
	for (int i=0; i < node_count; i++) {
		printf("Node %d: (%.1f, %.1f)\n", nodes[i].id, nodes[i].x, nodes[i].y);
	}
	printf("Loaded %d edges:\n", edge_count);
	for (int i=0; i < edge_count; i++) {
		printf("Edge %d: (%d, %d)\n", edges[i].id, edges[i].node1, edges[i].node2);
	}


	/* Close File */
	fclose(file);

	return 0;
}
