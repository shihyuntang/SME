.. _lfs:

Large File Server
=================

PySME does not come with all of the large atmosphere or NLTE grids
as part of the distribution. Instead Uppsala University provides
a server that serves the files when needed. Simply specify one of
the available filenames and PySME will fetch the file on your next run.

.. External hyperlinks, like `Python <http://www.python.org/>`_.

Available Files
---------------

Atmosphere grids:

  - recommended:

    - marcs2012.sav `(Gustafsson et al. 2008) <https://ui.adsabs.harvard.edu/abs/2008A%26A...486..951G>`_
    - marcs2012p_t0.0.sav
    - marcs2012p_t1.0.sav
    - marcs2012p_t2.0.sav
    - marcs2012s_t1.0.sav
    - marcs2012s_t2.0.sav
    - marcs2012s_t5.0.sav
    - marcs2012t00cooldwarfs.sav
    - marcs2012t01cooldwarfs.sav
    - marcs2012t02cooldwarfs.sav

  - deprecated:

    - atlas12.sav
    - atlas9_vmic0.0.sav
    - atlas9_vmic2.0.sav
    - interpatlas12.pro
    - interpmarcs2012.pro
    - ll_vmic2.0.sav

NLTE grids:

  - recommended:

    - marcs2012_Fe2016.grd `(Amarsi et al. 2016a) <https://ui.adsabs.harvard.edu/abs/2016MNRAS.463.1518A>`_
    - marcs2012_Li.grd `(Lind et al. 2009) <https://ui.adsabs.harvard.edu/abs/2009A%26A...503..541L>`_
    - marcs2012_Li2009.grd `(Lind et al. 2009) <https://ui.adsabs.harvard.edu/abs/2009A%26A...503..541L>`_
    - marcs2012_Mg2016.grd `(Osorio et al. 2016) <https://ui.adsabs.harvard.edu/abs/2016A%26A...586A.120O>`_
    - marcs2012_Na.grd `(Lind et al. 2011) <https://ui.adsabs.harvard.edu/abs/2011A%26A...528A.103L>`_
    - marcs2012_Na2011.grd `(Lind et al. 2011) <https://ui.adsabs.harvard.edu/abs/2011A%26A...528A.103L>`_
    - marcs2012_O2015.grd `(Amarsi et al. 2016b) <https://ui.adsabs.harvard.edu/abs/2016MNRAS.455.3735A>`_
    - marcs2012_Si2016.grd `(Amarsi & Asplund 2017) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.464..264A>`_
    - marcs2012p_t1.0_Ba.grd `(Mashonkina et al. 1999) <https://ui.adsabs.harvard.edu/abs/1999A%26A...343..519M>`_
    - marcs2012p_t1.0_Ca.grd `(Mashonkina et al. 2017) <https://ui.adsabs.harvard.edu/abs/2007A%26A...461..261M>`_
    - marcs2012p_t1.0_Na.grd `(Mashonkina et al. 2001) <https://ui.adsabs.harvard.edu/abs/2000ARep...44..790M>`_
    - marcs2012p_t1.0_O.grd `(Sitnova et al. 2013) <https://ui.adsabs.harvard.edu/abs/2013AstL...39..126S>`_
    - marcs2012s_t2.0_Ca.grd `(Mashonkina et al. 2007) <https://ui.adsabs.harvard.edu/abs/2007A%26A...461..261M>`_
    - marcs2012s_t2.0_Fe.grd `(Mashonkina et al. 2011) <https://ui.adsabs.harvard.edu/abs/2011A%26A...528A..87M>`_
    - marcs2012s_t2.0_Ti.grd `(Sitnova et al. 2016) <https://ui.adsabs.harvard.edu/abs/2016MNRAS.461.1000S>`_

  - deprecated:

    - nlte_Ba.grd
    - nlte_Ba_test.grd
    - nlte_C.grd
    - nlte_C_test.grd
    - nlte_Ca.grd
    - nlte_Ca_test.grd
    - nlte_Eu.grd
    - nlte_Eu_test.grd
    - nlte_Fe_multi_full.grd
    - nlte_Li_multi.grd
    - nlte_Na_multi_full.grd
    - nlte_Na_multi_sun.grd
    - nlte_Si.grd
    - nlte_Si_test.grd
