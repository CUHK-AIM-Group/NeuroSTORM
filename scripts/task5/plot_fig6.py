#!/usr/bin/env python
"""Draw Fig.6b from Task5's measured five-run Excel and raw Test predictions."""
import argparse
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from importlib.util import module_from_spec, spec_from_file_location
_task5_spec=spec_from_file_location("task5_demo",ROOT/"demo/task5.py")
_task5=module_from_spec(_task5_spec)
_task5_spec.loader.exec_module(_task5)
MODEL_NAMES=_task5.MODEL_NAMES
SEEDS=_task5.SEEDS
read_json=_task5.read_json
read_predictions=_task5.read_predictions
score_predictions=_task5.score_predictions
validate_protocol=_task5.validate_protocol

WIDTH,HEIGHT=1833,629
METHODS=MODEL_NAMES
BAR_COLORS=("#547178","#626F31","#643025","#AC861F","#99B3C0")
LEFT_COLORS=("#B4807B","#9EB0BA")
RIGHT_NOTE=("Bar height: mean value","Error bars: ±s.d.","5 independent runs")


def get_fonts():
    directory=Path(matplotlib.get_data_path())/"fonts/ttf"
    paths={}
    for key,name in (("regular","DejaVuSans.ttf"),("bold","DejaVuSans-Bold.ttf"),("italic","DejaVuSans-Oblique.ttf")):
        path=directory/name
        font_manager.fontManager.addfont(str(path))
        paths[key]=path
    return paths


def figure_rect(x,y,width,height):
    """Rectangles specified in pixels from the top-left at 100 dpi."""
    return [x/WIDTH,1-(y+height)/HEIGHT,width/WIDTH,height/HEIGHT]


def draw_panels(left_rows,accuracy_rows,fonts):
    fp=lambda size,bold=False:FontProperties(fname=str(fonts['bold' if bold else 'regular']),size=size)
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans',
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,
                         'axes.linewidth':0.9,'savefig.facecolor':'white'})
    # Avoid a binary-floating-point round-down to 1832 rather than 1833 raster pixels.
    figure=plt.figure(figsize=(np.nextafter(WIDTH/100,np.inf),np.nextafter(HEIGHT/100,np.inf)),dpi=100,facecolor='white')

    # Left panel: twenty Test predictions.
    field='P_true'
    arrays=[np.array([float(row[field]) for row in left_rows if int(row['label'])==label]) for label in (0,1)]
    n_people=len({row['person'] for row in left_rows})
    n_runs=len({row['seed'] for row in left_rows})
    left=figure.add_axes(figure_rect(147,134,768,370))
    bodies=left.violinplot(arrays,positions=[1,2],widths=.8,points=300,showextrema=False)['bodies']
    random=np.random.default_rng(20261101)
    for position,values,body,color in zip((1,2),arrays,bodies,LEFT_COLORS):
        body.set_facecolor(color);body.set_edgecolor('#747474');body.set_linewidth(1);body.set_alpha(.97)
        quantiles=left.hlines(np.quantile(values,[.25,.5,.75]),position-.4,position+.4,
                              color='#777777',linestyle='dotted',linewidth=1)
        quantiles.set_clip_path(body.get_paths()[0],left.transData)
        left.scatter(position+random.uniform(-.12,.12,len(values)),values,s=16,
                     color='#6f6f6f',alpha=.50,linewidths=0,zorder=4)
    # Expand only if needed to keep any supplied error point visible.
    bottom=min(.05,float(np.concatenate(arrays).min())-.025)
    left.set_xlim(.5,2.5);left.set_ylim(bottom,1.035)
    left.set_xticks([1,2],['Placebo','DMT-harmine'])
    left.set_yticks([.2,.4,.6,.8,1.0])
    left.tick_params(axis='both',direction='out',length=8,width=.8,pad=9)
    for label in left.get_xticklabels()+left.get_yticklabels():label.set_fontproperties(fp(22))
    left.set_title('Predicted state probability distribution',fontproperties=fp(24),pad=14)
    left.set_ylabel('Probability of correct state',fontproperties=fp(26),labelpad=9)
    left.set_xlabel('State (0 = placebo, 1 = DMT-harmine)',fontproperties=fp(24),labelpad=16)
    sample_note=left.text(.975,.045,f'{n_people} subjects × {n_runs} runs',transform=left.transAxes,
              ha='right',va='bottom',fontproperties=fp(21))
    figure.text(29/WIDTH,1-64/HEIGHT,'b',fontproperties=fp(36,True),va='baseline')
    figure.text(100/WIDTH,1-64/HEIGHT,'Medication state classification',fontproperties=fp(26),va='baseline')

    # Right panel: five seed accuracies, mean +/- sample standard deviation (ddof=1).
    values=np.array([[float(row[method]) for row in accuracy_rows] for method in METHODS])
    means=values.mean(axis=1);sd=values.std(axis=1,ddof=1)
    assert np.allclose(means,[statistics.mean(v) for v in values],atol=1e-15,rtol=0)
    assert np.allclose(sd,[statistics.stdev(v) for v in values],atol=1e-15,rtol=0)
    right=figure.add_axes(figure_rect(1053,95,670,447))
    right.bar(np.arange(5)-2,means,width=.9,color=BAR_COLORS,edgecolor='black',linewidth=1.1,
              yerr=sd,error_kw={'ecolor':'black','elinewidth':.9,'capsize':12,'capthick':.8},zorder=2)
    right.set_xlim(-2.82,2.82);right.set_ylim(.395,1.10)
    right.set_yticks([.4,.6,.8,1.0]);right.set_xticks([0],['DMT-HAR-MED'])
    right.tick_params(axis='y',direction='out',length=10,width=.8,pad=9)
    right.tick_params(axis='x',direction='out',length=11,width=.8,pad=11)
    for label in right.get_yticklabels():label.set_fontproperties(fp(22))
    for label in right.get_xticklabels():label.set_fontproperties(fp(22))
    right.spines['top'].set_visible(False);right.spines['right'].set_visible(False)

    # Explicit legend coordinates keep text and colored squares from shifting with data.
    patch_x=(993,1242,1381,1520,1640)
    text_x=(1028,1283,1422,1561,1680)
    legend_sizes=(21.0,)*len(METHODS)  # One font size for all five model names.
    right_text=[]
    for method,color,px,tx,size in zip(METHODS,BAR_COLORS,patch_x,text_x,legend_sizes):
        square=Rectangle((px/WIDTH,1-80/HEIGHT),28/WIDTH,28/HEIGHT,
                         transform=figure.transFigure,facecolor=color,edgecolor='black',linewidth=.8)
        figure.add_artist(square)
        text=figure.text(tx/WIDTH,1-76/HEIGHT,method,fontproperties=fp(size),va='baseline')
        text.set_gid('right_legend_'+method);right_text.append(text)

    # The narrow gray facet strip is an independent axis with all four borders.
    strip=figure.add_axes(figure_rect(1723,95,56,447),facecolor='#EBEBEB')
    strip.set_xticks([]);strip.set_yticks([])
    for spine in strip.spines.values():spine.set_color('black');spine.set_linewidth(.9)
    facet=strip.text(.5-2.5/56,.5+2/447,'Medication state classification',rotation=270,
                     ha='center',va='center',fontproperties=fp(20.25))
    facet.set_gid('right_facet');right_text.append(facet)

    # Summary labels for the five-run statistics.
    annotation=Rectangle((1411/WIDTH,1-215/HEIGHT),289/WIDTH,100/HEIGHT,
                         transform=figure.transFigure,facecolor='white',edgecolor='black',linewidth=.8,zorder=10)
    figure.add_artist(annotation)
    for number,(line,baseline) in enumerate(zip(RIGHT_NOTE,(148.3,179,207))):
        text=figure.text(1422/WIDTH,1-baseline/HEIGHT,line,va='baseline',ha='left',
                         fontproperties=fp(16.2),color='black',zorder=11)
        text.set_gid('right_note_line_'+str(number+1));right_text.append(text)

    figure.canvas.draw();renderer=figure.canvas.get_renderer()
    assert figure.canvas.get_width_height()==(WIDTH,HEIGHT),'Unexpected preview dimensions'
    sample_note.set_gid('left_subjects_times_runs')
    for text in right_text+[sample_note]:
        bounds=text.get_window_extent(renderer)
        assert bounds.x0>=0 and bounds.y0>=0 and bounds.x1<=WIDTH and bounds.y1<=HEIGHT,text.get_text()
    expected=list(METHODS)+['Medication state classification']+list(RIGHT_NOTE)
    assert [text.get_text() for text in right_text]==expected
    return figure,means,sd

def load_source(source,protocol):
    from openpyxl import load_workbook
    book=load_workbook(source,read_only=True,data_only=True)
    try:
        sheet=book["Accuracy"]
        models=[sheet.cell(1,c).value for c in range(2,7)]
        if len(set(models))!=5 or set(models)!=set(METHODS):
            raise ValueError("Accuracy sheet must contain all five DMT methods")
        accuracy=[dict(seed=seed) for seed in SEEDS]
        left=None
        for method in METHODS:
            column=models.index(method)+2
            rows=read_predictions(source.parent/"predictions"/(method+"_dmt.csv"))
            per_seed,summary=score_predictions(rows,protocol)
            for i,item in enumerate(per_seed):
                value=sheet.cell(i+2,column).value
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or abs(value-item["accuracy"])>1e-12:
                    raise ValueError(f"{method}: Excel and raw Test logits disagree; complete all five model commands")
                accuracy[i][method]=float(value)
            for row,key in ((7,"mean"),(8,"sample_SD")):
                value=sheet.cell(row,column).value
                if not isinstance(value,(int,float)) or abs(value-summary["accuracy"][key])>1e-12:
                    raise ValueError(f"{method}: Excel summary disagrees with the five runs")
            if method=="NeuroSTORM":left=rows
        return left,accuracy
    finally:
        book.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=Path("results/task5/figures"))
    parser.add_argument("--protocol",type=Path,default=ROOT/"demo/task5_protocol.json")
    parser.add_argument("--dpi",type=int,default=300)
    args=parser.parse_args(argv)
    if args.dpi<100:parser.error("--dpi must be at least 100")
    protocol=validate_protocol(read_json(args.protocol))
    left,accuracy=load_source(args.source.expanduser().resolve(),protocol)
    figure,means,sd=draw_panels(left,accuracy,get_fonts())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for extension in ("png","pdf","svg"):
        figure.savefig(args.output_dir/("Fig6b."+extension),dpi=100,facecolor="white")
    figure.savefig(args.output_dir/f"Fig6b_{args.dpi}dpi.png",dpi=args.dpi,facecolor="white")
    plt.close(figure)
    for model,mean,std in zip(METHODS,means,sd):
        print(f"{model}: Accuracy {mean*100:.2f}% ± {std*100:.2f}% s.d.")
    print(f"Figure saved in {args.output_dir}")


if __name__=="__main__":
    main()
