"""
OpenVocal: Open-source vocal pitch correction with visual blob editing.

A minimalist, open-source vocal editing platform designed to replicate
the core "graphical pitch correction" workflow of Melodyne.
"""

__version__ = "0.1.0"
__author__ = "OpenVocal Contributors"

from openvocal.models.note_blob import NoteBlob
from openvocal.models.project import Project

__all__ = ["NoteBlob", "Project", "__version__"]
