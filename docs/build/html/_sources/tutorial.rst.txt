Tutorial
============

``Sivio`` is a high-level OO Python package which aims to provide an easy and intuitive way of interacting with nearby Bluetooth Low Energy (BLE) devices (GATT servers). In essence, this package is an extension of the ``bluepy`` package created by Ian Harvey (see `here <https://github.com/IanHarvey/bluepy/>`_)

The aim here was to define a single object which would allow users to perform the various operations performed by the ``bluepy.btle.Peripheral``, ``bluepy.btle.Scanner``, ``bluepy.btle.Service`` and ``bluepy.btle.Characteristic`` classes of ``bluepy``, from one central place. This functionality is facilitated by the ``simpleble.SimpleBleClient`` and ``simpleble.SimpleBleDevice`` classes, where the latter is an extention/subclass of ``bluepy.btle.Peripheral``, combined with properties of ``bluepy.btle.ScanEntry``.

The current implementation has been developed in Python 3 and tested on a Raspberry Pi Zero W, running Raspbian 9 (stretch), but should work with Python 2.7+ (maybe with minor modifications in terms of printing and error handling) and most Debian based OSs.
