"""Use normal desktop D-Bus activation so PDF viewers do not confine Videos.

Requires the local Ubuntu codecs prepared in ~/.local/share/jeweltracker/ubuntu-codecs.
This configures the current user's Videos launcher; no AppArmor rules are changed.
"""
from pathlib import Path
import shutil
import subprocess


def configure():
    home = Path.home()
    codecs = home / '.local/share/jeweltracker/ubuntu-codecs'
    wrapper = codecs / 'videos-with-codecs'
    if not wrapper.is_file():
        raise SystemExit('Local video decoder launcher is missing: ' + str(wrapper))
    services = home / '.local/share/dbus-1/services'
    services.mkdir(parents=True, exist_ok=True)
    service = services / 'org.gnome.Totem.service'
    if service.exists() and not (codecs / 'original-user-Totem.service').exists():
        shutil.copy2(service, codecs / 'original-user-Totem.service')
    service.write_text('[D-BUS Service]\nName=org.gnome.Totem\n'
                       f'Exec={wrapper} --gapplication-service\n')
    applications = home / '.local/share/applications'
    applications.mkdir(parents=True, exist_ok=True)
    original = Path('/usr/share/applications/org.gnome.Totem.desktop').read_text()
    # Keep the application ID consistent with the D-Bus service name.
    # Exec is only a fallback; capable callers activate org.gnome.Totem over D-Bus.
    desktop = original.replace('Exec=totem ', f'Exec={wrapper} ')
    if 'DBusActivatable=true' not in desktop:
        raise SystemExit('Unexpected Videos desktop entry: D-Bus activation missing')
    (applications / 'org.gnome.Totem.desktop').write_text(desktop)
    obsolete = applications / 'jeweltracker-videos.desktop'
    if obsolete.is_file() and str(wrapper) in obsolete.read_text():
        obsolete.unlink()
    subprocess.run(['update-desktop-database', str(applications)], check=True)
    subprocess.run(['xdg-mime', 'default', 'org.gnome.Totem.desktop',
                    'video/x-msvideo', 'video/mp4'], check=True)
    subprocess.run(['dbus-send', '--session', '--print-reply',
                    '--dest=org.freedesktop.DBus', '/org/freedesktop/DBus',
                    'org.freedesktop.DBus.ReloadConfig'], check=True, capture_output=True)
    print('Videos now uses D-Bus activation with the local Ubuntu decoders.')


if __name__ == '__main__':
    configure()
