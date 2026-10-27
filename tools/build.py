import os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src/page.html(Claude 아티팩트용 원본) → index.html(GitHub Pages용 완성본)
src=open(os.path.join(ROOT,'src/page.html'),encoding='utf-8').read()
a=src.index('<style>'); b=src.index('</style>')+8
headbits=src[:a]; style=src[a:b]; body=src[b:]
headbits=headbits.replace('<meta name="claude:viewport-fit-sides" content="cover">\n','')
leaflet='<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" crossorigin="anonymous">\n<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js" crossorigin="anonymous"></script>\n'
out=('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
 '<meta name="description" content="기능상속 — 사라지는 지역 생활서비스를 기존 자원에 이어주는 AI 지역상생 서비스 (2026 AI 라이프 솔루션 챌린지 시연용 프로토타입)">\n'
 '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n'
 +headbits+leaflet+style+'\n</head>\n<body>\n'+body.lstrip()+'\n</body>\n</html>\n')
open(os.path.join(ROOT,'index.html'),'w',encoding='utf-8').write(out); print(len(out))
