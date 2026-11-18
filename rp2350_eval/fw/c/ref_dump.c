/* Debug aid: host reference dump of a kernel's working memory (pairs with rp2350-eval --dump). */
#include <stdio.h>
#include <stdlib.h>
#include "kernels.h"
int main(int argc, char **argv)
{
    unsigned ws;
    if (argc < 4) return 2;
    kernel_run((unsigned)atoi(argv[1]), (unsigned)atoi(argv[2]), 2, &ws);
    { FILE *f = fopen(argv[3], "wb"); fwrite(ev_arena, 1, ws, f); fclose(f); }
    return 0;
}
