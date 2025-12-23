
import site
block_cipher = None

from PyInstaller.utils.hooks import collect_submodules

a = Analysis(
    ['easyClock_v3.1.py'],
    pathex=[site.getsitepackages()[0]],
    binaries=[],
    datas=[
        ('easyClock.icns', '.')
    ],
    hiddenimports=collect_submodules('matplotlib')+
                  collect_submodules('scipy') +
                  collect_submodules('statsmodels') +
                  collect_submodules('pywt'),
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='easyClock',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False
)

app = BUNDLE(
    exe,
    name='easyClock.app',
    icon='easyClock.icns',
    bundle_identifier='com.easyclock.app'
)
