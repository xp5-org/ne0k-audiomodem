/* Host (x86) reference: prints "id param checksum" lines the harness compares with the
 * M33's results. Built natively from the same sources: bit-identical or it is a finding. */
#include <stdio.h>
#include <stdlib.h>
#include "kernels.h"
int main(int argc, char **argv)
{
    int i;
    for (i = 1; i + 1 < argc; i += 2) {
        unsigned id = (unsigned)atoi(argv[i]), p = (unsigned)atoi(argv[i + 1]), ws;
        unsigned c = kernel_run(id, p, 2, &ws);
        printf("%u %u %08x %u\n", id, p, c, ws);
    }
    return 0;
}
