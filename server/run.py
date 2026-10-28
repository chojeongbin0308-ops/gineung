"""서버 + 공개 주소(Cloudflare 임시 터널)를 함께 켠다.  실행: python run.py
주소가 나오면 클립보드에 복사하고 관리자 페이지를 연다(붙여넣고 저장만 하면 됨)."""
import os, re, sys, subprocess, threading, time, webbrowser, shutil, urllib.parse
HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.getenv('GINEUNG_PORT', '8787'))
SITE = 'https://chojeongbin0308-ops.github.io/gineung/'


def find_cloudflared():
    local = os.path.join(HERE, 'bin', 'cloudflared.exe' if os.name == 'nt' else 'cloudflared')
    return local if os.path.exists(local) else shutil.which('cloudflared')


def main():
    server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', str(PORT), '--log-level', 'warning'], cwd=HERE)
    cf = find_cloudflared()
    if not cf:
        print('[주의] cloudflared가 없어 공개 주소 없이 이 컴퓨터에서만 켜집니다. setup_server.bat를 다시 실행해 주세요.')
        server.wait(); return
    tunnel = subprocess.Popen([cf, 'tunnel', '--no-autoupdate', '--url', f'http://127.0.0.1:{PORT}'], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace')
    url = None
    for line in tunnel.stderr:
        m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
        if m and not url:
            url = m.group(0)
            open(os.path.join(HERE, 'current_url.txt'), 'w', encoding='utf-8').write(url)
            print('\n' + '=' * 60 + f'\n  공개 주소: {url}\n' + '=' * 60)
            if os.name == 'nt':
                subprocess.run('clip', input=url, text=True, shell=True)
                print('  주소를 클립보드에 복사했습니다.')
            print('  관리자 페이지가 열리면 [서버 주소 저장]을 누르세요. 이 창은 닫지 마세요.\n')
            time.sleep(8)  # 모델·터널이 준비될 시간
            webbrowser.open(SITE + '?server=' + urllib.parse.quote(url, safe='') + '#admin')
    print('[터널 종료]')
    server.terminate()


if __name__ == '__main__':
    main()
