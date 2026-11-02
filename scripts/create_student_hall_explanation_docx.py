#!/usr/bin/env python3
"""Readable Korean Word explanation using the bundled document runtime.

Preset: standard_business_brief; header: memo_masthead (no rule).
Named overrides: KoreanFont=Apple SD Gothic Neo; Title=26pt navy;
TableText=10pt/1.15; Caption=9pt/1.15; Note=10pt/1.15 gray fill.
"""
from pathlib import Path
import json,sys
from PIL import Image,ImageDraw,ImageFont
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

SKILL=Path('/Users/baekkyunghwan/.codex/plugins/cache/openai-primary-runtime/documents/26.520.11634/skills/documents')
sys.path.insert(0,str(SKILL/'scripts'))
from table_geometry import apply_table_geometry

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/student_hall_explanation';OUT.mkdir(parents=True,exist_ok=True)
DEST=ROOT/'docs/학생회관_3D복원_과정과결과_정리.docx'
EXP=ROOT/'experiments/student_hall_lidar'
FONT='Apple SD Gothic Neo';BLUE='2E74B5';NAVY='0B2545';MUTED='596575'
URL='https://stereo-reconstruction-lab-kyunghwan.sillycod2.chatgpt.site/student-hall.html'

def flow_figure():
    im=Image.new('RGB',(1800,820),'white');d=ImageDraw.Draw(im)
    fp='/System/Library/Fonts/AppleSDGothicNeo.ttc'
    font=ImageFont.truetype(fp,36);small=ImageFont.truetype(fp,30)
    def box(x,y,w,h,title,sub,fill):
        d.rounded_rectangle((x,y,x+w,y+h),radius=16,fill=fill,outline='#B8C7D5',width=2)
        for text,yy,f in [(title,y+23,font),(sub,y+70,small)]:
            b=d.textbbox((0,0),text,font=f);d.text((x+(w-(b[2]-b[0]))/2,yy),text,font=f,fill='#17324E')
    def arrow(a,b):
        d.line((a,b),fill='#607D98',width=5)
        import math
        ang=math.atan2(b[1]-a[1],b[0]-a[0]);r=17
        d.polygon([b,(b[0]-r*math.cos(ang-.5),b[1]-r*math.sin(ang-.5)),(b[0]-r*math.cos(ang+.5),b[1]-r*math.sin(ang+.5))],fill='#607D98')
    box(20,20,830,125,'추가 입력: 원시 LiDAR와 보정값','LiDAR 스캔 48개 + 센서 간 위치·방향','#E8EEF5')
    box(950,20,830,125,'공통 입력: 학생회관 RGB 사진·기존 COLMAP','사진 48장 + 카메라·특징점 정보','#E8EEF5')
    box(20,255,830,125,'구조 복원: 스캔 정합과 경로 보정','사진·COLMAP을 초기값으로 활용, LiDAR를 연결','#F4F6F9')
    box(950,255,830,125,'외관 학습: Gaussian Splatting 후보 비교','같은 사진 분할로 COLMAP·LiDAR 초기화 비교','#F4F6F9')
    box(20,490,830,125,'사진 색상 부여 → 점군·표면 생성','전체 지도 중 사진에서 확인한 영역에 색상 부여','#E8EEF5')
    box(950,490,830,125,'평가 화질이 가장 좋은 후보 선택','원래 COLMAP 기반 12,000회 학습 모델','#E8EEF5')
    box(20,685,830,115,'사이트 상단: 3D 구조 탐색','컬러 점군 / 건물 표면 / LiDAR 전체','#F4F6F9')
    box(950,685,830,115,'사이트 하단: 렌더링 영상','사진 사이 시점에서 본 별도의 지역 모델','#F4F6F9')
    arrow((435,145),(435,255));arrow((1200,145),(600,255));arrow((1365,145),(1365,255))
    arrow((435,380),(435,490));arrow((1365,380),(1365,490));arrow((435,615),(435,685));arrow((1365,615),(1365,685))
    im.save(OUT/'workflow.png')

def element(tag,**attrs):
    e=OxmlElement('w:'+tag)
    for k,v in attrs.items():e.set(qn('w:'+k),str(v))
    return e

def style_font(style,size,color='202833',bold=False):
    style.font.name=FONT;style.font.size=Pt(size);style.font.color.rgb=RGBColor.from_string(color);style.font.bold=bold
    rpr=style.element.get_or_add_rPr();fonts=rpr.find(qn('w:rFonts'))
    if fonts is None:fonts=element('rFonts');rpr.append(fonts)
    for key in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+key),FONT)

def setup(doc):
    sec=doc.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.top_margin=sec.bottom_margin=sec.left_margin=sec.right_margin=Inches(1)
    sec.header_distance=sec.footer_distance=Inches(.492)
    styles=doc.styles
    for name,size,color,bold,before,after in [('Normal',11,'202833',False,0,6),('Title',26,NAVY,True,0,8),
        ('Subtitle',12,MUTED,False,0,8),('Heading 1',16,BLUE,True,16,8),('Heading 2',13,BLUE,True,12,6),('Heading 3',12,'1F4D78',True,8,4)]:
        s=styles[name];style_font(s,size,color,bold);pf=s.paragraph_format
        pf.space_before=Pt(before);pf.space_after=Pt(after);pf.line_spacing=1.1
        pf.widow_control=True
        if name.startswith('Heading'):pf.keep_with_next=True
    for name,size,color,after,line in [('TableText',10,'202833',0,1.15),('CaptionText',9,MUTED,6,1.15),
        ('Metadata',9,MUTED,3,1.1),('NoteText',10,NAVY,8,1.15),('TableCitation',9,MUTED,4,1.1),('BulletText',11,'202833',8,1.167)]:
        s=styles.add_style(name,1);s.base_style=styles['Normal'];style_font(s,size,color)
        s.paragraph_format.space_before=Pt(4 if name=='TableCitation' else 0);s.paragraph_format.space_after=Pt(after);s.paragraph_format.line_spacing=line
        s.paragraph_format.widow_control=True
    # Real Word bullet definition with preset marker/text/hanging geometry.
    numbering=doc.part.numbering_part.element
    abstract=element('abstractNum',abstractNumId=90);abstract.append(element('multiLevelType',val='singleLevel'))
    lvl=element('lvl',ilvl=0);lvl.extend([element('start',val=1),element('numFmt',val='bullet'),element('lvlText',val='•'),element('lvlJc',val='left')])
    pp=element('pPr');tabs=element('tabs');tabs.append(element('tab',val='num',pos=720));pp.append(tabs);pp.append(element('ind',left=720,hanging=360));lvl.append(pp);abstract.append(lvl);numbering.append(abstract)
    num=element('num',numId=90);num.append(element('abstractNumId',val=90));numbering.append(num)
    p=sec.header.paragraphs[0];p.style=styles['Metadata'];p.text='RECONSTRUCTION LAB  |  학생회관 복원 설명';p.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    p=sec.footer.paragraphs[0];p.style=styles['Metadata'];p.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    p.add_run('학생회관 3D 복원  ·  ')
    field=element('fldSimple',instr=' PAGE ');run=element('r');text=element('t');text.text='1';run.append(text);field.append(run);p._p.append(field)
    settings=doc.settings.element;settings.append(element('updateFields',val='true'))
    doc.core_properties.title='학생회관 3D 복원: 입력 자료, 처리 과정과 결과'
    doc.core_properties.subject='RGB·COLMAP·LiDAR 구조 복원과 Gaussian Splatting 렌더링의 구분'
    doc.core_properties.author='';doc.core_properties.keywords='학생회관, COLMAP, LiDAR, 3D reconstruction'

def para(doc,text,style=None):
    p=doc.add_paragraph(style=style)
    for i,part in enumerate(text.split('**')):p.add_run(part).bold=(i%2==1)
    return p

def bullet(doc,text):
    p=para(doc,text,'BulletText');pr=p._p.get_or_add_pPr();np=element('numPr');np.extend([element('ilvl',val=0),element('numId',val=90)]);pr.append(np)

def note(doc,text):
    p=para(doc,text,'NoteText');pr=p._p.get_or_add_pPr();pr.append(element('shd',fill='F4F6F9'));pr.append(element('ind',left=120,right=120));return p

def link(doc,label,url):
    p=doc.add_paragraph();h=OxmlElement('w:hyperlink');h.set(qn('r:id'),doc.part.relate_to(url,RT.HYPERLINK,is_external=True))
    r=element('r');rp=element('rPr');rp.append(element('color',val=BLUE));rp.append(element('u',val='single'));r.append(rp);t=element('t');t.text=label;r.append(t);h.append(r);p._p.append(h);return p

def table(doc,headers,rows,widths,selected=None):
    t=doc.add_table(rows=1+len(rows),cols=len(headers));t.autofit=False
    for i,row in enumerate([headers]+rows):
        trpr=t.rows[i]._tr.get_or_add_trPr();trpr.append(element('cantSplit'))
        if i==0:trpr.append(element('tblHeader',val=1))
        for j,text in enumerate(row):
            c=t.cell(i,j);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p=c.paragraphs[0];p.style=doc.styles['TableText'];r=p.add_run(str(text));r.bold=(i==0)
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER if j>0 and len(str(text))<22 else WD_ALIGN_PARAGRAPH.LEFT
            if i==0 or i-1==selected:c._tc.get_or_add_tcPr().append(element('shd',fill='E8EEF5' if i==0 else 'F4F6F9'))
    borders=element('tblBorders')
    for side in ['top','left','bottom','right','insideH','insideV']:borders.append(element(side,val='single',sz=4,color='CCD5DF'))
    t._tbl.tblPr.append(borders);apply_table_geometry(t,widths,table_width_dxa=9360,indent_dxa=120)
    para(doc,'','TableCitation');return t

def image(doc,path,width=6.5,alt=''):
    p=doc.add_paragraph();p.paragraph_format.keep_with_next=True
    pic=p.add_run().add_picture(str(path),width=Inches(width));pic._inline.docPr.set('descr',alt)
    return p

def paired_images(doc,left,right,caption):
    p=doc.add_paragraph();p.paragraph_format.keep_with_next=True
    for i,path in enumerate([left,right]):
        if i:p.add_run('  ')
        pic=p.add_run().add_picture(str(path),width=Inches(3.15));pic._inline.docPr.set('descr',caption.split(' / ')[min(i,1)])
    para(doc,caption,'CaptionText')

def page(doc,title):doc.add_page_break();doc.add_heading(title,level=1)

def main():
    flow_figure();doc=Document();setup(doc)
    # Page 1: a substantive overview, not an empty decorative cover.
    doc.add_heading('학생회관 3D 복원\n과정과 결과 정리',level=0)
    para(doc,'어떤 자료를 사용했고, 어떤 방법으로 무엇을 만들었는가','Subtitle')
    para(doc,'문서 작성: 2026.10.22  |  실험 기준: 2026.10.21  |  촬영 구간: 01-26-15-41-24','Metadata')
    doc.add_heading('먼저 구분해야 할 두 결과',level=1)
    para(doc,'현재 사이트에는 **공간의 형태를 보여주는 구조 모델**과 **사진처럼 보이게 학습한 렌더링 모델**이 함께 있다. 두 모델은 목적과 최종 선택 방법이 다르다.')
    bullet(doc,'**구조 모델:** 원시 LiDAR 48개를 사진·기존 COLMAP 정보와 함께 정합해 지도와 표면을 만들었다.')
    bullet(doc,'**렌더링 모델:** 사진으로 외관을 학습하는 Gaussian Splatting 후보를 비교하고, 평가 화질이 가장 좋은 COLMAP 기반 모델을 선택했다.')
    image(doc,OUT/'workflow.png',alt='사진과 COLMAP 및 LiDAR 입력에서 구조 모델과 렌더링 모델이 별도로 만들어지는 흐름도')
    para(doc,'그림 1. 공통 입력에서 출발하지만, 구조 지도와 사진형 렌더링은 별도로 생성·평가한 결과이다.','CaptionText')
    note(doc,'**자료의 출처:** 친구가 만든 완성 LiDAR 지도는 사용하지 않았다. 프로젝트에 이미 있던 같은 촬영의 원시 LiDAR 자료로 지도를 직접 만들었다.')
    link(doc,'공유 사이트: 학생회관 3D 복원 결과 열기',URL)

    page(doc,'입력 자료와 기존 COLMAP 결과')
    para(doc,'대상 장면은 **조형물이 있는 중정 → 넓은 계단·경사로 → 학생회관 출입구**로 이동하는 구간이다. 프로젝트에 받아두었던 Pixel-aligned RGB-NIR Stereo 데이터셋의 촬영 자료를 사용했다.')
    table(doc,['입력','내용','용도'],[
        ['RGB 사진','왼쪽 카메라 사진 48장','같은 지점 찾기, 색상 부여, 외관 학습'],
        ['원시 LiDAR','시점별 스캔 48개','거리와 공간 구조 확보'],
        ['센서 보정값','렌즈·센서 간 위치와 방향','사진과 LiDAR 좌표 연결'],
        ['기존 COLMAP','카메라 위치와 3D 특징점','초기값 및 비교 기준']], [2050,3150,4160])
    doc.add_heading('기존 COLMAP은 무엇을 계산했나',level=2)
    para(doc,'COLMAP은 여러 사진에서 같은 창문 모서리나 바닥 무늬를 찾고, 그 대응관계로 **사진을 찍은 카메라의 위치·방향**과 **해당 지점의 3D 위치**를 함께 추정한다. 이렇게 얻은 점을 특징점이라고 한다.')
    para(doc,'기존 학생회관 결과는 두 덩어리로 나뉘어 있었다. 합계는 **중복을 제외한 카메라 40개, 특징점 5,183개**였다. 이는 벽과 바닥 전체가 채워진 표면보다, 공간 곳곳에 특징점이 흩어진 성긴 복원에 가깝다. 사진만으로 얻은 결과의 크기를 실제 미터 단위에 맞추는 과정도 필요했다.')
    p=doc.add_paragraph();p.paragraph_format.keep_with_next=True
    for i in [0,26,47]:
        pic=p.add_run().add_picture(str(EXP/'images'/f'{i:06d}.png'),width=Inches(2.12));pic._inline.docPr.set('descr',f'원본 학생회관 사진 프레임 {i}')
        if i!=47:p.add_run(' ')
    para(doc,'그림 2. 왼쪽부터 중정(프레임 0), 계단·경사로(26), 학생회관 표지판이 있는 입구(47).','CaptionText')
    note(doc,'**이전 결과와의 구분:** 이번 학생회관 구간은 이전 사이트의 긴 09-05-17-34-36 경로와 다른 촬영이다. 이번 주된 처리에는 RGB 왼쪽 사진과 LiDAR를 사용했다. 앞서 수행한 RGB·NIR 좌우 스테레오 실험과 구분해야 한다.')

    page(doc,'LiDAR 지도를 만든 처리 과정')
    para(doc,'LiDAR 스캔 하나는 그 순간 센서 위치를 기준으로 측정한 주변의 3D 점들이다. 움직이며 얻은 스캔들을 하나의 공간에 놓으려면, 각 시점 사이의 이동과 회전을 찾아야 한다. 이를 정합이라고 한다.')
    for title,text in [
        ('사진과 LiDAR의 좌표 연결','제공된 보정값으로 사진의 렌즈 왜곡을 보정하고, LiDAR 점이 사진의 어느 픽셀에 해당하는지 계산했다. 사진 위의 계단·벽과 측정점이 대응하는지 확인했다.'),
        ('COLMAP의 크기를 미터 단위로 보정','COLMAP이 추정한 점의 깊이와 같은 방향의 LiDAR 깊이를 비교했다. 깊이 비율을 이용해 기존 복원의 크기를 보정했다. 이는 미터 단위를 부여하는 과정이며, 실제 치수 정확도를 검증한 것은 아니다.'),
        ('여러 이동 추정에서 시작해 세밀하게 정합','LiDAR 단독 KISS-ICP, 사진 특징점과 LiDAR 깊이를 이용한 PnP, 기존 COLMAP의 상대 위치를 각각 초기값으로 비교했다. 이후 ICP로 겹치는 벽·바닥의 위치와 방향을 큰 간격부터 작은 간격까지 조정했다.'),
        ('큰 회전 구간의 시작 위치를 다시 탐색','방향을 크게 바꾸면 초기 추정이 틀려 ICP가 잘못 맞을 수 있다. 이런 구간은 FPFH로 공간 구조의 특징을 비교하고 RANSAC으로 잘못된 후보를 걸러, 위치와 방향부터 다시 찾았다.'),
        ('전체 이동 경로를 함께 보정','인접 시점과 일부 건너뛴 시점의 정합 관계를 포즈 그래프로 묶고 함께 최적화했다. 최종 구조에는 이 경로를 사용해 48개 시점을 하나의 좌표계에 연결했다.')]:
        doc.add_heading(title,level=2);para(doc,text)
    note(doc,'**공동 최적화도 별도로 시도했다.** COLMAP 카메라·특징점과 LiDAR 표면을 함께 보정한 후보는 사진의 재투영 일치도가 좋아졌지만, 스캔끼리의 일관성은 떨어졌다. 따라서 최종 구조 모델에는 포즈 그래프 결과를 남겼다. Colmap-PCD 공식 프로그램 전체를 설치·재현한 실험은 아니다.')

    page(doc,'사진 색상 부여와 건물 표면 생성')
    para(doc,'정합한 스캔을 하나의 지도에 모은 뒤, 반복해서 관측한 가까운 점을 묶고 잡음을 일부 제거했다. 이어 각 3D 점을 사진에 투영하고 가려짐·깊이를 비교해 **해당 사진의 색상**을 가져왔다. 사진에서 색상을 확인하지 못한 구조는 회색으로 남겼다.')
    table(doc,['결과물','원본 규모','숫자가 의미하는 것'],[
        ['전체 LiDAR 지도','1,648,179개 점','사진 밖의 주변 구조까지 포함'],
        ['컬러 점군','232,136개 점','전체 지도 중 사진 색상을 붙인 부분'],
        ['선택한 Poisson 표면','2,120,479개 삼각형','측정점 사이에 추정해 만든 면']], [2450,2650,4260])
    para(doc,'전체 LiDAR 점과 컬러 점은 더해서 세는 숫자가 아니다. **전체 지도 중 일부에 색상이 붙은 것**이다. 또한 기존 5,183개는 사진으로 계산한 특징점이고, 새 컬러 점은 색상을 붙인 LiDAR 점이다. 점 수 증가를 정확도 향상 배수로 해석하면 안 된다.')
    doc.add_heading('점 사이를 면으로 연결한 방법',level=2)
    para(doc,'여러 깊이를 누적하는 TSDF와, 점의 위치·표면 방향으로 이어지는 면을 추정하는 Poisson을 비교했다. 더 넓은 구조를 연결한 Poisson을 선택했다. 초기 후보의 큰 가짜 면을 실제 시점에서 확인한 뒤, 센서 가까이 약 3m 이내 반환점과 관측점에서 멀리 떨어진 표면 정점을 제외했다.')
    paired_images(doc,EXP/'images/000026.png',EXP/'poisson_view_26.jpg','그림 3. 왼쪽: 원본 사진 / 오른쪽: 복원한 표면을 같은 시점에서 본 영상. 회색·구멍·식생 주변의 뭉개짐이 남아 있다.')
    para(doc,'공유 사이트는 빠르게 열리도록 영역별 표면을 최대 약 18만 개 삼각형으로 줄여 표시한다. 원본 메시와 사이트 표시용 메시의 규모는 다르다.','CaptionText')

    page(doc,'사진처럼 보이는 렌더링의 생성과 선택')
    para(doc,'Gaussian Splatting은 카메라 위치와 초기 3D 점을 바탕으로 색상 덩어리들을 배치하고, 렌더링한 영상이 실제 사진과 비슷해지도록 학습하는 방법이다. 학습 후에는 사진 사이의 시점에서도 영상을 만들 수 있다. **위의 LiDAR 표면과 별도로 학습한 외관 모델**이다.')
    para(doc,'계단·입구의 프레임 20~45, 총 26장 중 **21장은 외관 학습**, **5장은 평가**에 사용했다. 평가 사진은 20·26·32·38·44번이다. 다만 카메라 위치 추정에는 전체 사진을 사용했으므로, 모든 처리에서 완전히 숨겨둔 사진은 아니다.')
    table(doc,['초기 카메라·점','학습 횟수','평가 PSNR ↑'],[
        ['원래 COLMAP 카메라·특징점','12,000회','23.28 dB'],
        ['공동 보정 카메라·COLMAP 점','12,000회','22.48 dB'],
        ['공동 보정 카메라·LiDAR 점','12,000회','15.78 dB'],
        ['원래 COLMAP 모델을 추가 학습','30,000회','23.17 dB']], [5500,1800,2060],selected=0)
    para(doc,'PSNR은 평가 사진과 렌더링이 얼마나 비슷한지 보는 화질 지표이며 높을수록 좋다. 가장 좋은 **원래 COLMAP 기반 12,000회 모델**을 최종 영상으로 선택했다. 더 오래 학습한 후보와 LiDAR 점으로 초기화한 후보는 이번 조건에서 더 좋은 화질을 내지 못했다.')
    paired_images(doc,EXP/'gs_images/000032.png',EXP/'gs_colmap_12000/heldout_02.png','그림 4. 왼쪽: 외관 학습에서 제외한 원본 프레임 32 / 오른쪽: 선택한 Gaussian 모델의 같은 시점 렌더링.')
    note(doc,'이번 학습은 RGB 손실을 사용했다. LiDAR 깊이가 맞도록 학습 내내 강제하는 깊이 손실이나 LI-GS 논문 전체를 구현한 결과는 아니다. LiDAR의 구조 복원 효과와 렌더링 화질 효과를 구분해야 한다.')

    page(doc,'개선 수치의 의미와 현재 한계')
    para(doc,'구조 비교에서는 같은 37개 인접 시점 쌍에 동일한 LiDAR 스캔을 배치했다. 기존 COLMAP 경로를 사용할 때와 새 경로를 사용할 때, 스캔끼리 얼마나 잘 겹치는지 비교했다. ICP 정합에 쓰지 않은 반환점 668,519개로 일관성을 검사했다.')
    table(doc,['경로 구성','연결 시점','25cm 이내 겹침 ↑'],[
        ['원본 COLMAP + 크기 보정','40 / 48','71.71%'],
        ['LiDAR 단독 KISS-ICP','48 / 48','70.96%'],
        ['전역 정합 + 포즈 그래프','48 / 48','77.81%'],
        ['COLMAP·LiDAR 공동 보정','48 / 48','74.69%']], [5350,1750,2260],selected=2)
    para(doc,'71.71% → 77.81%는 **서로 다른 시점의 LiDAR 관측이 더 일관되게 연결되었다**는 뜻이다. 기존 COLMAP 특징점 몇 개가 맞았는지를 세는 점수가 아니며, 건물이 실제로 몇 cm 정확하게 복원됐다는 뜻도 아니다.')
    doc.add_heading('확인한 개선',level=2)
    bullet(doc,'기존 40개 카메라가 포함된 성긴 사진 복원에서 출발해, LiDAR 기준으로 48개 시점을 하나의 경로에 연결했다.')
    bullet(doc,'사진만으로 얻은 특징점에 더해 거리 기반 구조, 사진 색상이 붙은 점군, 건물 표면을 생성했다.')
    bullet(doc,'회전 구간의 잘못된 정합과 표면의 큰 가짜 차폐 면을 발견하고 수정했다. 후보별 지표를 비교해 구조와 렌더링을 각각 선택했다.')
    doc.add_heading('아직 확인하지 못한 것과 남은 문제',level=2)
    bullet(doc,'별도로 측량한 건물 치수·기준점이 없어 절대 위치와 크기 오차를 확정하지 못했다. LiDAR 자체도 정답으로 간주하지 않았다.')
    bullet(doc,'지붕·실내·모든 외벽이 빠짐없이 포함된 모델이 아니다. 회색 영역, 구멍, 잔여 잡음, 부정확한 난간·식생·유리 구조가 남아 있다.')
    bullet(doc,'Gaussian 후보 비교는 난수까지 통제한 다중 반복 통계 실험이 아니다. 이전 09-05 경로의 PSNR과는 장면·평가 사진이 달라 직접 비교할 수 없다.')
    note(doc,'현재 결과는 **학생회관을 여러 관측에서 더 일관되게 연결한 부분 복원**으로 해석해야 한다. 사진처럼 보이는 것과 실제 구조가 정확한 것은 별도로 검증할 항목이다.')

    page(doc,'사이트 읽는 법과 주요 용어')
    link(doc,'학생회관 결과 사이트 열기',URL)
    table(doc,['화면·버튼','보여주는 결과'],[
        ['원본 COLMAP','기존 특징점. 비교용으로 LiDAR 크기 보정 및 배치'],
        ['개선된 컬러 점군','새로 정합한 LiDAR 지도 중 사진 색상을 붙인 영역'],
        ['LiDAR 전체','사진 밖 주변을 포함한 전체 거리 기반 지도'],
        ['건물 표면','점 사이를 Poisson으로 연결한 메시'],
        ['하단 렌더링 영상','별도로 선택한 COLMAP 기반 Gaussian 모델']], [2650,6710])
    para(doc,'먼저 계단·입구 범위에서 원본 COLMAP과 컬러 점군을 번갈아 보고, 건물 표면으로 전환해 공간의 연결 상태를 살펴본다. 아래 영상과 원본 사진을 비교하면 외관 모델의 선명함과 뭉개지는 부분을 확인할 수 있다.')
    doc.add_heading('주요 용어',level=2)
    for term,meaning in [
        ('점군','3D 좌표를 가진 점들의 모음. 아직 점 사이에 면이 없을 수 있다.'),
        ('특징점','여러 사진에서 같은 지점으로 대응시켜 3D 위치를 계산한 점.'),
        ('포즈','카메라나 센서의 위치와 바라보는 방향.'),
        ('정합','서로 다른 관측을 같은 공간에서 겹치도록 위치·방향을 맞추는 작업.'),
        ('메시','점 사이를 삼각형 등으로 연결해 만든 표면.'),
        ('재투영','3D 점이 카메라 사진의 어느 픽셀에 들어오는지 계산하는 것.'),
        ('렌더링','3D 표현을 특정 시점에서 바라본 2D 영상으로 만드는 것.')]:para(doc,f'**{term}:** {meaning}')
    doc.add_heading('근거와 기록',level=2)
    para(doc,'수치와 처리 과정은 프로젝트의 STUDENT_HALL_LIDAR_REPORT.md, evaluation.json, fused/summary.json 및 Gaussian 후보별 metrics.json을 대조해 정리했다. 이 문서는 기존 실험을 설명하는 자료이며 새로운 학습 결과를 추가한 것은 아니다.','CaptionText')
    link(doc,'전체 실험 기록과 재실행 명령 보기',URL.replace('student-hall.html','student-hall-report.md'))
    link(doc,'원본 데이터: Pixel-aligned RGB-NIR Stereo · CVPR 2025','https://github.com/divisonofficer/Pixel_aligned_RGB_NIR_Stereo')
    para(doc,'데이터 출처: Jinnyeong Kim · Seung-Hwan Baek / POSTECH. 친구의 완성 지도는 사용하지 않았다.','CaptionText')
    doc.save(DEST)
    # Structural preset audit (render QA is performed separately).
    assert round(doc.sections[0].page_width.inches,2)==8.5
    assert round(doc.sections[0].left_margin.inches,2)==1.
    for t in doc.tables:
        widths=[int(e.get(qn('w:w'))) for e in t._tbl.tblGrid]
        assert sum(widths)==9360
        assert t._tbl.tblPr.find(qn('w:tblW')).get(qn('w:w'))=='9360'
        assert t._tbl.tblPr.find(qn('w:tblInd')).get(qn('w:w'))=='120'
        for row in t.rows:
            assert [int(c._tc.get_or_add_tcPr().find(qn('w:tcW')).get(qn('w:w'))) for c in row.cells]==widths
    assert len(doc.inline_shapes)==8
    audit={'preset':'standard_business_brief','header':'memo_masthead (no rule)',
        'named_overrides':['KoreanFont','Title','TableText','CaptionText','NoteText'],
        'planned_pages':7,'tables':len(doc.tables),'images':len(doc.inline_shapes),'paragraphs':len(doc.paragraphs),
        'page_and_table_geometry':'passed','visual_render_QA':'unavailable: LibreOffice/soffice is not installed; canonical renderer attempted',
        'deliverable':str(DEST)}
    (OUT/'structural_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    print(json.dumps(audit,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
