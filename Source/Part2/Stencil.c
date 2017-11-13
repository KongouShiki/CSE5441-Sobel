/***
 * File: Stencil.h
 */

#include "Stencil.h"

void Stencil_MoveRight(Stencil_t *stencil)
{
   stencil->top++;
   stencil->middle++;
   stencil->bottom++;
}

void Stencil_MoveToNextRow(Stencil_t *stencil)
{
   /*
    * Since the array is contiguous and the stencil is at the rightmost
    * boundary, the next row wraps around and is three elements ahead of
    * the current element
    */
   stencil->top += 3;
   stencil->middle += 3;
   stencil->bottom += 3;
}
