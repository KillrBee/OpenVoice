"""Entry point for OpenVocal application."""

import sys


def main() -> int:
    """Main entry point for the OpenVocal application."""
    from PyQt6.QtWidgets import QApplication
    from openvocal.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("OpenVocal")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("OpenVocal")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
