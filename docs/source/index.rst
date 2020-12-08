.. sivio documentation master file, created by
   sphinx-quickstart on Tue Dec  1 15:41:58 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Sivio Documentation
=================================

Ionospherically contaminated visibilities simulator; This tool simulates different ionospheric conditions and produces the corrupted sky visibilities.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Introduction
================================

.. toctree::
   :maxdepth: 2

   intro

Installation
------------------------------------------
Grab the source code from this git repo:

``git clone https://github.com/kariukic/sivio``

Then navigate into that directory and run

``pip install .``

or alternatively

``python setup.py install``

Usage
------------------------------------------
Check out the tutorial page for an example.

.. toctree::
   :maxdepth: 2

   tutorial

Modules
------------------------------------------

.. toctree::
   :maxdepth: 2

   modules

Acknowledgements
-------------------
For full funtionality ``Sivio`` makes use of several other astronomy software tools such as Cthulhu_, WSClean_ and Aegean_. 

.. _Cthulhu: https://gitlab.com/chjordan/cthulhu
.. _WSClean: https://sourceforge.net/p/wsclean/wiki/Installation/
.. _Aegean: https://ui.adsabs.harvard.edu/abs/2012MNRAS.422.1812H/abstract


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
