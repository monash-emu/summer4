# Multi-property force-of-infection mixing

## Problem

summer2 can attach a mixing matrix to each stratification and combines them
(Kronecker-style) so contacts respect every mixing axis at once.

summer4's `ForceOfInfection.group_by` (`src/summer4/epi/infection.py`) takes a
**single** `Property`. After `FlowModel.stratify` adds another property, mixing
across the new axis is homogeneous: the FOI still groups only by the original
`group_by` property and sums over the rest.

## Why it hurts

Age × location (or age × strain) models that need assortative mixing on both
axes cannot express a product contact structure without collapsing one axis.

## Done when

`ForceOfInfection.group_by` accepts several properties and a product
(or explicitly constructed) mixing matrix whose axes match that product.
