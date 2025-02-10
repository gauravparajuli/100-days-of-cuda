#include <stdio.h>

// double pointer stores the address of another pointer

int main() {
    int **ptr; // ptr is pointer to a pointer to an integer

    int x = 10;
    int *p = &x; // p points to x
    int **pp = &p; // pp points to pointer p

    printf("%d\n", **pp) // 10
    
    return 0;
}


