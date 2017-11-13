/***
 * File: Stencil.h
 * Desc: Data type to optimize extracting a 3x3 stencil from a larger double matrix
 */

 #ifndef STENCIL_H
 #define STENCIL_H

typedef struct
{
   double *top;
   double *middle;
   double *bottom;
} Stencil_t;

/*
 * Move a stencil one element to the right in a matrix.
 *
 * @precondition -- stencil right is not at the rightmost column of the matrix
 * @param stencil -- the stencil
 */
void Stencil_MoveRight(Stencil_t &stencil);

/*
 * Move a stencil at the rightmost column of a matrix to the beginning of
 * the next row.
 *
 * @precondition -- stencil right is at the rightmost column of the matrix
 * @precondition -- stencil bottom is not at the last row of the matrix
 * @param stencil -- the stencil
 */
void Stencil_MoveToNextRow(Stencil_t &stencil);

 #endif
