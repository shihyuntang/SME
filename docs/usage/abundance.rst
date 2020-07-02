.. _abund:

Abundance
=========

The abundance contains information about the overall
metallicity of the star as well as the individual
abundances of each element.
SME can simulate and fit the first 99 elements of
the periodic table (from Hydrogen H to Einsteinium Es).

Abundance formats
-----------------

PySME supports a variety of formats to input the abundance.
However they are all internally converted to the 'H=12'
type before use.

:H=12:
    Abundance values are log10 of the fraction of nuclei of
    each element in any form relative to the number of hydrogen
    in any form plus an offset of 12. For the Sun, the nuclei
    abundance values of H, He, and Li are approximately 12,
    10.9, and 1.05.

:n/nTot:
    Abundance values are the fraction of nuclei
    of each element in any form relative to the total for all
    elements in any form. For the Sun, the abundance values of
    H, He, and Li are approximately 0.92, 0.078, and 1.03e-11.

:n/nH:
    Abundance values are the fraction of nuclei
    of each element in any form relative to the number of
    hydrogen nuclei in any form. For the Sun, the abundance
    values of H, He, and Li are approximately 1, 0.085, and
    1.12e-11.

:sme:
    For hydrogen, the abundance value is the fraction of all
    nuclei that are hydrogen, including all ionization states
    and treating molecules as constituent atoms. For the other
    elements, the abundance values are log10 of the fraction of
    nuclei of each element in any form relative to the total for
    all elements in any form. For the Sun, the abundance values
    of H, He, and Li are approximately 0.92, -1.11, and -11.0.

:Fe=12:
    Abundance values are log10 of the fraction of nuclei of
    each element in any form relative to the number of iron
    in any form plus an offset of 12. For the Sun, the nuclei
    abundance values of H, He, and Li are approximately 16.5,
    15.43, and 5.55.

:n/nFe:
    Abundance values are the fraction of nuclei
    of each element in any form relative to the number of
    iron nuclei in any form. For the Sun, the abundance
    values of H, He, and Li are approximately 3.16e4, 2.69e3, and
    3.55e-7.

Solar metallicity
-----------------

PySME contains three pre defined sets of solar abundances,
for you to choose from. They are:

:asplund2009:
    Asplund, Grevesse, Sauval, Scott (2009,  Annual Review of Astronomy
    and Astrophysics, 47, 481)

:grevesse2007:
    Grevesse, Asplund, Sauval (2007, Space Science Review, 130, 105)

:lodders2003:
    Lodders 2003 (ApJ, 591, 1220)

They can be initialized by passing their name during the
creation of the abundance. E.g. Abund("grevesse2007").
The default solar abundance is 'asplund2009' and is also
available using Abund.solar().
