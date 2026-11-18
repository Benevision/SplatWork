"""Warped pseudo-views (EVPGS fine stage, with sensor depth): each chosen training photo is re-drawn from a camera
moved along gravity, using its own LiDAR depth; hidden and unreliable pixels are masked."""
import sys, json, os, numpy as np, cv2
from pathlib import Path
R=Path(sys.argv[1]); split=Path(sys.argv[2]); out=Path(sys.argv[3]); shifts=[float(x) for x in sys.argv[4].split(',')]; every=int(sys.argv[5])
W=1400
ds=json.loads((split/'dataset.json').read_text()); up=np.array(ds.get('gravityUp') or ds['upVector'],float); up/=np.linalg.norm(up)
out.mkdir(parents=True,exist_ok=True); (out/'pseudo').mkdir(exist_ok=True)
for f in R.iterdir():
    if f.name!='dataset.json' and not (out/f.name).exists(): os.symlink(f,out/f.name)
train=[c for c in ds['cameras'] if not c.get('holdout')]
cams=list(ds['cameras']); nid=max(c['id'] for c in cams)+1; kept=0; cover=[]
for k,c in enumerate(train):
    if k%every: continue
    s=W/c['width']; H=int(round(c['height']*s)); fx,fy,cx,cy=c['fx']*s,c['fy']*s,c['cx']*s,c['cy']*s
    img=cv2.resize(cv2.imread(str(R/c['image'])),(W,H),interpolation=cv2.INTER_AREA)
    d=np.fromfile(R/c['depth'],dtype=np.float16).astype(np.float32).reshape(c['depthHeight'],c['depthWidth'])
    k3=np.ones((3,3),np.uint8); rng=(cv2.dilate(d,k3)-cv2.erode(d,k3))/np.maximum(d,1e-3)
    ok=((d>0.05)&(rng<0.06)).astype(np.float32)
    D=cv2.resize(d,(W,H),interpolation=cv2.INTER_LINEAR); OK=cv2.resize(ok,(W,H),interpolation=cv2.INTER_LINEAR)>0.99
    c2w=np.array(c['cameraToWorld'],float); c2w=np.vstack([c2w,[0,0,0,1]]) if c2w.shape[0]==3 else c2w
    u,v=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(H,dtype=np.float32))
    for sh in shifts:
        p2w=c2w.copy(); p2w[:3,3]+=up*sh; w2p=np.linalg.inv(p2w); w2s=np.linalg.inv(c2w)
        # forward: source depth -> pseudo depth (z-buffer, 2x2 footprint)
        X=np.stack([(u-cx)/fx*D,(v-cy)/fy*D,D],-1)[OK]; Xw=X@c2w[:3,:3].T+c2w[:3,3]; Xp=Xw@w2p[:3,:3].T+w2p[:3,3]
        z=Xp[:,2]; g=z>0.05; Xp=Xp[g]; z=z[g]
        up_=Xp[:,0]/z*fx+cx; vp=Xp[:,1]/z*fy+cy
        Dt=np.full((H,W),np.inf,np.float32)
        for du in (0,1):
            for dv in (0,1):
                iu=np.floor(up_).astype(int)+du; iv=np.floor(vp).astype(int)+dv; m=(iu>=0)&(iu<W)&(iv>=0)&(iv<H)
                np.minimum.at(Dt,(iv[m],iu[m]),z[m])
        have=np.isfinite(Dt); Dt0=np.where(have,Dt,0)
        # backward: pseudo pixel -> source pixel, with a depth agreement test (hidden surfaces fail it)
        Y=np.stack([(u-cx)/fx*Dt0,(v-cy)/fy*Dt0,Dt0],-1); Yw=Y@p2w[:3,:3].T+p2w[:3,3]; Ys=Yw@w2s[:3,:3].T+w2s[:3,3]
        zs=np.maximum(Ys[...,2],1e-3); us=(Ys[...,0]/zs*fx+cx).astype(np.float32); vs=(Ys[...,1]/zs*fy+cy).astype(np.float32)
        col=cv2.remap(img,us,vs,cv2.INTER_LINEAR); Dsrc=cv2.remap(D,us,vs,cv2.INTER_LINEAR); oks=cv2.remap(OK.astype(np.float32),us,vs,cv2.INTER_LINEAR)
        valid=have&(us>=1)&(us<W-2)&(vs>=1)&(vs<H-2)&(oks>0.99)&(np.abs(zs-Dsrc)<0.03*Dsrc)
        valid=cv2.erode(valid.astype(np.uint8),np.ones((5,5),np.uint8))>0
        if valid.mean()<0.25: continue
        name=f'pseudo/{nid:06d}'
        cv2.imwrite(str(out/(name+'.jpg')),np.where(valid[...,None],col,0),[cv2.IMWRITE_JPEG_QUALITY,95])
        cv2.imwrite(str(out/(name+'_mask.png')),(valid*255).astype(np.uint8))
        dl=cv2.resize(np.where(valid,Dt0,0),(c['depthWidth'],c['depthHeight']),interpolation=cv2.INTER_NEAREST)
        dl.astype(np.float16).tofile(out/(name+'.bin'))
        n=dict(c); n.update(id=nid,image=name+'.jpg',mask=name+'_mask.png',depth=name+'.bin',width=W,height=H,fx=fx,fy=fy,cx=cx,cy=cy,
                 cameraToWorld=p2w[:len(c['cameraToWorld'])].tolist(),holdout=False,pseudo=True)
        cams.append(n); nid+=1; kept+=1; cover.append(valid.mean())
meta=dict(ds); meta['cameras']=cams; (out/'dataset.json').write_text(json.dumps(meta))
print(f'{kept} pseudo-views, mean valid area {np.mean(cover):.0%}, train photos {len(train)}')
