#!/usr/bin/env python3
"""
Build script for creating executable distributions of Robotic Car Simulation

Supports PyInstaller, cx_Freeze, and other packaging tools for creating
standalone executables across different platforms.
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
import json
import platform

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from version import __version__


class ExecutableBuilder:
    """Builder for creating executable distributions"""
    
    def __init__(self, build_tool: str = "pyinstaller"):
        self.build_tool = build_tool
        self.project_root = Path(__file__).parent.parent
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.spec_file = self.project_root / "robotic_car_sim.spec"
        
        # Platform info
        self.platform = platform.system().lower()
        self.architecture = platform.machine().lower()
        
    def clean_build_dirs(self):
        """Clean previous build directories"""
        print("Cleaning build directories...")
        
        for directory in [self.build_dir, self.dist_dir]:
            if directory.exists():
                shutil.rmtree(directory)
                print(f"Removed {directory}")
                
    def create_pyinstaller_spec(self):
        """Create PyInstaller spec file"""
        spec_content = f'''
# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

block_cipher = None

# Data files to include
datas = [
    ('src/ui/*.ui', 'ui'),
    ('src/assets/*', 'assets'),
    ('src/shaders/*', 'shaders'),
    ('requirements.txt', '.'),
]

# Hidden imports
hiddenimports = [
    'PyQt6.QtCore',
    'PyQt6.QtWidgets',
    'PyQt6.QtOpenGL',
    'PyQt6.QtGui',
    'numpy',
    'psutil',
    'sqlite3',
]

a = Analysis(
    ['main.py'],
    pathex=[str(Path.cwd())],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='RoboticCarSimulation',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='build_scripts/version_info.txt',
    icon='assets/icon.ico' if Path('assets/icon.ico').exists() else None,
)
'''
        
        with open(self.spec_file, 'w') as f:
            f.write(spec_content)
            
        print(f"Created PyInstaller spec file: {self.spec_file}")
        
    def create_version_info(self):
        """Create version info file for Windows builds"""
        version_info_content = f'''
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({__version__.replace('.', ', ')}, 0),
    prodvers=({__version__.replace('.', ', ')}, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [StringStruct(u'CompanyName', u'Robotic Car Simulation Team'),
           StringStruct(u'FileDescription', u'Robotic Car Simulation'),
           StringStruct(u'FileVersion', u'{__version__}'),
           StringStruct(u'InternalName', u'RoboticCarSimulation'),
           StringStruct(u'LegalCopyright', u'Copyright (C) 2024'),
           StringStruct(u'OriginalFilename', u'RoboticCarSimulation.exe'),
           StringStruct(u'ProductName', u'Robotic Car Simulation'),
           StringStruct(u'ProductVersion', u'{__version__}')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
        
        version_file = self.project_root / "build_scripts" / "version_info.txt"
        version_file.parent.mkdir(exist_ok=True)
        
        with open(version_file, 'w') as f:
            f.write(version_info_content)
            
        print(f"Created version info file: {version_file}")
        
    def build_with_pyinstaller(self, onefile: bool = True):
        """Build executable using PyInstaller"""
        print("Building with PyInstaller...")
        
        # Create spec file
        self.create_pyinstaller_spec()
        
        # Create version info for Windows
        if self.platform == "windows":
            self.create_version_info()
            
        # Build command
        cmd = ["pyinstaller"]
        
        if onefile:
            cmd.append("--onefile")
        else:
            cmd.append("--onedir")
            
        cmd.extend([
            "--windowed",  # No console window
            "--clean",
            str(self.spec_file)
        ])
        
        # Execute build
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode == 0:
            print("PyInstaller build completed successfully!")
            return True
        else:
            print("PyInstaller build failed!")
            return False    
        
    def build_with_cx_freeze(self):
        """Build executable using cx_Freeze"""
        print("Building with cx_Freeze...")
        
        # Create setup script for cx_Freeze
        setup_content = f'''
import sys
from cx_Freeze import setup, Executable
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

# Build options
build_options = {{
    "packages": ["PyQt6", "numpy", "psutil"],
    "excludes": ["tkinter", "matplotlib"],
    "include_files": [
        ("src/ui/", "ui/"),
        ("src/assets/", "assets/"),
        ("src/shaders/", "shaders/"),
    ],
    "zip_include_packages": ["*"],
    "zip_exclude_packages": [],
}}

# Executable configuration
base = None
if sys.platform == "win32":
    base = "Win32GUI"  # Use for GUI applications on Windows

executables = [
    Executable(
        "main.py",
        base=base,
        target_name="RoboticCarSimulation",
        icon="assets/icon.ico" if Path("assets/icon.ico").exists() else None
    )
]

setup(
    name="RoboticCarSimulation",
    version="{__version__}",
    description="Robotic Car Simulation",
    options={{"build_exe": build_options}},
    executables=executables
)
'''
        
        setup_file = self.project_root / "setup_cx_freeze.py"
        with open(setup_file, 'w') as f:
            f.write(setup_content)
            
        # Build command
        cmd = [sys.executable, str(setup_file), "build"]
        
        result = subprocess.run(cmd, cwd=self.project_root)
        
        # Clean up
        setup_file.unlink()
        
        if result.returncode == 0:
            print("cx_Freeze build completed successfully!")
            return True
        else:
            print("cx_Freeze build failed!")
            return False
            
    def create_installer(self):
        """Create installer package"""
        print("Creating installer package...")
        
        if self.platform == "windows":
            return self.create_windows_installer()
        elif self.platform == "darwin":
            return self.create_macos_installer()
        elif self.platform == "linux":
            return self.create_linux_installer()
        else:
            print(f"Installer creation not supported for platform: {self.platform}")
            return False
            
    def create_windows_installer(self):
        """Create Windows installer using NSIS or Inno Setup"""
        print("Creating Windows installer...")
        
        # Check if NSIS is available
        try:
            subprocess.run(["makensis", "/VERSION"], capture_output=True, check=True)
            return self.create_nsis_installer()
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("NSIS not found, skipping installer creation")
            return False
            
    def create_nsis_installer(self):
        """Create NSIS installer script and build"""
        nsis_script = f'''
!define APPNAME "Robotic Car Simulation"
!define COMPANYNAME "Robotic Car Simulation Team"
!define DESCRIPTION "Advanced autonomous vehicle simulation platform"
!define VERSIONMAJOR 1
!define VERSIONMINOR 0
!define VERSIONBUILD 0
!define HELPURL "https://github.com/roboticarsim/robotic-car-simulation"
!define UPDATEURL "https://github.com/roboticarsim/robotic-car-simulation/releases"
!define ABOUTURL "https://github.com/roboticarsim/robotic-car-simulation"
!define INSTALLSIZE 500000

RequestExecutionLevel admin
InstallDir "$PROGRAMFILES\\${{COMPANYNAME}}\\${{APPNAME}}"
Name "${{APPNAME}}"
Icon "assets\\icon.ico"
outFile "dist\\RoboticCarSimulation-{__version__}-Setup.exe"

!include LogicLib.nsh

page components
page directory
page instfiles

!macro VerifyUserIsAdmin
UserInfo::GetAccountType
pop $0
${{If}} $0 != "admin"
    messageBox mb_iconstop "Administrator rights required!"
    setErrorLevel 740
    quit
${{EndIf}}
!macroend

function .onInit
    setShellVarContext all
    !insertmacro VerifyUserIsAdmin
functionEnd

section "install"
    setOutPath $INSTDIR
    file /r "dist\\RoboticCarSimulation\\*"
    
    writeUninstaller "$INSTDIR\\uninstall.exe"
    
    createDirectory "$SMPROGRAMS\\${{COMPANYNAME}}"
    createShortCut "$SMPROGRAMS\\${{COMPANYNAME}}\\${{APPNAME}}.lnk" "$INSTDIR\\RoboticCarSimulation.exe" "" "$INSTDIR\\RoboticCarSimulation.exe"
    createShortCut "$DESKTOP\\${{APPNAME}}.lnk" "$INSTDIR\\RoboticCarSimulation.exe" "" "$INSTDIR\\RoboticCarSimulation.exe"
    
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "DisplayName" "${{APPNAME}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "UninstallString" "$\\"$INSTDIR\\uninstall.exe$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "QuietUninstallString" "$\\"$INSTDIR\\uninstall.exe$\\" /S"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "InstallLocation" "$\\"$INSTDIR$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "DisplayIcon" "$\\"$INSTDIR\\RoboticCarSimulation.exe$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "Publisher" "${{COMPANYNAME}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "HelpLink" "${{HELPURL}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "URLUpdateInfo" "${{UPDATEURL}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "URLInfoAbout" "${{ABOUTURL}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "DisplayVersion" "{__version__}"
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "VersionMajor" ${{VERSIONMAJOR}}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "VersionMinor" ${{VERSIONMINOR}}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "NoModify" 1
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "NoRepair" 1
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "EstimatedSize" ${{INSTALLSIZE}}
sectionEnd

section "uninstall"
    delete "$INSTDIR\\uninstall.exe"
    rmDir /r "$INSTDIR"
    
    delete "$SMPROGRAMS\\${{COMPANYNAME}}\\${{APPNAME}}.lnk"
    rmDir "$SMPROGRAMS\\${{COMPANYNAME}}"
    delete "$DESKTOP\\${{APPNAME}}.lnk"
    
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}"
sectionEnd
'''
        
        nsis_file = self.project_root / "installer.nsi"
        with open(nsis_file, 'w') as f:
            f.write(nsis_script)
            
        # Build installer
        cmd = ["makensis", str(nsis_file)]
        result = subprocess.run(cmd, cwd=self.project_root)
        
        # Clean up
        nsis_file.unlink()
        
        return result.returncode == 0
        
    def create_macos_installer(self):
        """Create macOS installer package"""
        print("Creating macOS installer...")
        
        # Create .app bundle structure
        app_name = "RoboticCarSimulation.app"
        app_path = self.dist_dir / app_name
        
        # Create bundle directories
        contents_dir = app_path / "Contents"
        macos_dir = contents_dir / "MacOS"
        resources_dir = contents_dir / "Resources"
        
        for directory in [contents_dir, macos_dir, resources_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create Info.plist
        info_plist = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>RoboticCarSimulation</string>
    <key>CFBundleIdentifier</key>
    <string>com.roboticarsim.simulation</string>
    <key>CFBundleName</key>
    <string>Robotic Car Simulation</string>
    <key>CFBundleVersion</key>
    <string>{__version__}</string>
    <key>CFBundleShortVersionString</key>
    <string>{__version__}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>'''
        
        with open(contents_dir / "Info.plist", 'w') as f:
            f.write(info_plist)
            
        print(f"Created macOS app bundle: {app_path}")
        return True
        
    def create_linux_installer(self):
        """Create Linux installer (AppImage or .deb)"""
        print("Creating Linux installer...")
        
        # Try to create AppImage
        try:
            return self.create_appimage()
        except Exception as e:
            print(f"AppImage creation failed: {e}")
            
        # Fallback to creating .deb package
        try:
            return self.create_deb_package()
        except Exception as e:
            print(f"DEB package creation failed: {e}")
            
        return False
        
    def create_appimage(self):
        """Create AppImage for Linux"""
        print("Creating AppImage...")
        
        # This would require linuxdeploy and other AppImage tools
        # For now, just create a basic directory structure
        
        appdir = self.dist_dir / "RoboticCarSimulation.AppDir"
        appdir.mkdir(exist_ok=True)
        
        # Create desktop file
        desktop_content = f'''[Desktop Entry]
Type=Application
Name=Robotic Car Simulation
Exec=RoboticCarSimulation
Icon=robotic-car-simulation
Categories=Education;Science;
'''
        
        with open(appdir / "robotic-car-simulation.desktop", 'w') as f:
            f.write(desktop_content)
            
        print(f"Created AppImage directory: {appdir}")
        return True
        
    def create_deb_package(self):
        """Create Debian package"""
        print("Creating DEB package...")
        
        # Create package structure
        package_name = f"robotic-car-simulation_{__version__}_amd64"
        package_dir = self.dist_dir / package_name
        
        # Create DEBIAN directory
        debian_dir = package_dir / "DEBIAN"
        debian_dir.mkdir(parents=True, exist_ok=True)
        
        # Create control file
        control_content = f'''Package: robotic-car-simulation
Version: {__version__}
Section: education
Priority: optional
Architecture: amd64
Depends: python3 (>= 3.9), python3-pyqt6
Maintainer: Robotic Car Simulation Team <team@roboticarsim.com>
Description: Advanced autonomous vehicle simulation platform
 A comprehensive simulation environment for developing and testing
 autonomous vehicle algorithms with realistic physics and AI systems.
'''
        
        with open(debian_dir / "control", 'w') as f:
            f.write(control_content)
            
        print(f"Created DEB package structure: {package_dir}")
        return True
        
    def build(self, tool: str = None, onefile: bool = True, create_installer: bool = False):
        """Main build method"""
        if tool is None:
            tool = self.build_tool
            
        print(f"Starting build process with {tool}...")
        print(f"Platform: {self.platform}")
        print(f"Architecture: {self.architecture}")
        print(f"Version: {__version__}")
        
        # Clean previous builds
        self.clean_build_dirs()
        
        # Build executable
        success = False
        if tool == "pyinstaller":
            success = self.build_with_pyinstaller(onefile)
        elif tool == "cx_freeze":
            success = self.build_with_cx_freeze()
        else:
            print(f"Unknown build tool: {tool}")
            return False
            
        if not success:
            print("Build failed!")
            return False
            
        # Create installer if requested
        if create_installer:
            installer_success = self.create_installer()
            if installer_success:
                print("Installer created successfully!")
            else:
                print("Installer creation failed, but executable build succeeded.")
                
        print("Build process completed!")
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Build executable for Robotic Car Simulation")
    
    parser.add_argument(
        "--tool", 
        choices=["pyinstaller", "cx_freeze"], 
        default="pyinstaller",
        help="Build tool to use"
    )
    
    parser.add_argument(
        "--onedir", 
        action="store_true",
        help="Create one-directory distribution instead of one-file"
    )
    
    parser.add_argument(
        "--installer", 
        action="store_true",
        help="Create installer package"
    )
    
    parser.add_argument(
        "--clean-only", 
        action="store_true",
        help="Only clean build directories"
    )
    
    args = parser.parse_args()
    
    builder = ExecutableBuilder(args.tool)
    
    if args.clean_only:
        builder.clean_build_dirs()
        return
        
    success = builder.build(
        tool=args.tool,
        onefile=not args.onedir,
        create_installer=args.installer
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()