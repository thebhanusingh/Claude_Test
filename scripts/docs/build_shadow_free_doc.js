const fs=require('fs'), path=require('path');
const {Document,Packer,Paragraph,TextRun,ImageRun,HeadingLevel,AlignmentType,Table,TableRow,TableCell,WidthType,ShadingType,BorderStyle,LevelFormat,Footer,PageNumber,PageBreak}=require('docx');
const IMG=process.argv[2], OUT=process.argv[3];
const sizeOf=f=>{const b=fs.readFileSync(f);let i=2;while(i<b.length){if(b[i]!==0xFF){i++;continue;}const m=b[i+1];const L=b.readUInt16BE(i+2);if(m>=0xC0&&m<=0xC3)return{h:b.readUInt16BE(i+5),w:b.readUInt16BE(i+7)};i+=2+L;}};
const W=624; // 6.5 in at 96 dpi
const p=(t,o={})=>new Paragraph({spacing:{after:120},...o,children:runs(t)});
// **bold** markup
function runs(t){return t.split(/(\*\*[^*]+\*\*)/).filter(s=>s).map(s=>s.startsWith('**')?new TextRun({text:s.slice(2,-2),bold:true}):new TextRun(s));}
const h1=t=>new Paragraph({heading:HeadingLevel.HEADING_1,pageBreakBefore:true,children:[new TextRun(t)]});
const h2=t=>new Paragraph({heading:HeadingLevel.HEADING_2,children:[new TextRun(t)]});
const bl=t=>new Paragraph({numbering:{reference:'b',level:0},spacing:{after:60},children:runs(t)});
function fig(file,label,cap,maxw=W){const f=path.join(IMG,file);const s=sizeOf(f);let w=Math.min(maxw,W),h=Math.round(w*s.h/s.w);if(h>560){h=560;w=Math.round(h*s.w/s.h);}
 return [new Paragraph({alignment:AlignmentType.CENTER,keepNext:true,spacing:{before:120,after:60},children:[new ImageRun({type:'jpg',data:fs.readFileSync(f),transformation:{width:w,height:h},altText:{title:label,description:cap,name:file}})]}),
 new Paragraph({spacing:{after:200},children:[new TextRun({text:label+'. ',bold:true,size:19}),new TextRun({text:cap,italics:true,size:19})]})];}
const bd={style:BorderStyle.SINGLE,size:4,color:'BBBBBB'};
function table(cols,rows){const tw=cols.reduce((a,b)=>a+b,0);const cell=(t,i,hd)=>new TableCell({width:{size:cols[i],type:WidthType.DXA},borders:{top:bd,bottom:bd,left:bd,right:bd},margins:{top:60,bottom:60,left:100,right:100},shading:hd?{type:ShadingType.CLEAR,fill:'E7EEF6',color:'auto'}:undefined,children:[new Paragraph({children:[new TextRun({text:t,bold:hd,size:20})]})]});
 return new Table({width:{size:tw,type:WidthType.DXA},columnWidths:cols,rows:rows.map((r,j)=>new TableRow({tableHeader:j===0,children:r.map((t,i)=>cell(t,i,j===0))}))});}
const sp=()=>new Paragraph({spacing:{after:120},children:[]});

const c=[];
// Title page
c.push(new Paragraph({spacing:{before:2400,after:240},alignment:AlignmentType.CENTER,children:[new TextRun({text:'Removing baked-in sun shadows from a Gaussian splat',bold:true,size:44})]}));
c.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:480},children:[new TextRun({text:'The full process, stage by stage, with test images',size:28,color:'555555'})]}));
c.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:120},children:[new TextRun({text:'Test scene: IMG_6556, a chair on grass in direct sun',size:24})]}));
c.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{after:120},children:[new TextRun({text:'Status as of 13 October 2026: work in progress, still being tested',size:24})]}));
c.push(new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:960},children:[new TextRun({text:'Written for a reader who is not a Gaussian splatting specialist. Every number here is recorded, with its settings, in the project lab notes (docs/lab-notes.md).',italics:true,size:20,color:'555555'})]}));

// Background
c.push(h1('1. Background'));
c.push(h2('What a Gaussian splat is'));
c.push(p('A **Gaussian splat** is a way of storing a real 3D scene that was filmed on video. Instead of a mesh made of triangles, the scene is made of millions of small, soft, coloured blobs called **Gaussians**. Each blob has a position, a size and stretch in three directions, an orientation, a colour and a transparency. When the scene is viewed from any angle, the blobs blend into an image that looks like a photograph.'));
c.push(p('The test scene has about **2.9 million** blobs. The chair alone is about **24,000**.'));
c.push(h2('The problem'));
c.push(p('The blobs store the colours the camera saw, **including the shadows**. The shadow under the chair is painted onto the grass blobs, and a shaded slat of the chair is painted dark. When the scene is used in Blender or Unreal Engine with new lighting, those painted-in shadows look wrong, because the new light comes from a different direction.'));
c.push(h2('The idea'));
c.push(p('Fully modelling materials and the physics of light is hard and slow. This project treats shadow removal as **repainting** instead: find which blobs are in shadow, then recolour each one using sunlit blobs on the **same surface**, following the direction of the surface texture (its "grain"). The method assumes one sun and one main object, and adds no time to the normal splat-making process beyond a few minutes of processing.'));
c.push(...fig('s1_scene.jpg','Figure 1.1','The test scene, one frame of the phone video. The chair\'s shadow falls on the grass, and the upper backrest is shaded by the chair itself.',300));

// Overview
c.push(h1('2. The five stages at a glance'));
c.push(p('The project brief splits the work into five stages. Steps that go beyond the brief are marked **[Addition]** in this document and in the lab notes.'));
c.push(table([2100,4060,2200],[
 ['Stage','What it does','Status'],
 ['1. Capture and 3D reconstruction','Turn a phone video into a splat, with the camera position for every frame.','Done'],
 ['2. Sun direction','Find the sun\'s angle from the shape of the cast shadows.','Done: 55.1° vs true 55.5°'],
 ['3. Surfaces and grain','Split the object\'s blobs into surfaces and find each surface\'s texture direction.','Done'],
 ['4. Which blobs are in shadow','Flag blobs the sun cannot reach, checked against the photos.','Done, with additions'],
 ['5. Repainting the shadows','Recolour shadowed blobs from sunlit blobs on the same surface, along the grain.','First test done; being refined'],
]));

// Stage 1
c.push(h1('3. Stage 1: Capture and 3D reconstruction'));
c.push(p('**What:** turn a phone video into a splat.'));
c.push(p('**How:** about 650 of the sharpest frames are picked from the video. A method called **Structure from Motion** (the program COLMAP) matches features between the photos and works out where the camera was for each one. A splat is then trained (with the program nerfstudio) so that its images match the photos as closely as possible.'));
c.push(p('**Result:** a splat of 2.9 million blobs. It is kept in COLMAP\'s coordinate system, so every later stage can line up a 3D blob with the exact pixel it came from in each original photo. This link between the 3D blobs and the photos is used throughout the later stages.'));

// Stage 2
c.push(h1('4. Stage 2: Estimating the sun direction'));
c.push(p('**What:** find where the sun was, as an elevation angle above the ground and a compass direction.'));
c.push(p('**Why:** stage 4 needs the sun direction to decide which blobs the sun could reach.'));
c.push(p('**How, in plain words:** for a guessed sun direction, the chair\'s shadow is cast onto the ground in 3D, like holding a torch over a model. That predicted shadow is compared with where the photos actually show shadow. Many directions are tried, and the best match wins. As the brief requires, the sun is found from the shadow geometry, not from the blobs\' colour values.'));
c.push(p('To know where the photos show shadow, an existing neural network called **SDDNet** marks the shadow pixels in each photo. Each ground blob is looked up in every photo that sees it, and each photo gives a "shadow" or "lit" vote.'));
c.push(p('**Checking the answer:** the video\'s time and GPS location give the true sun elevation, 55.51° (accurate to about ±0.1°). The compass direction cannot be checked, because the 3D scene has no built-in north.'));
c.push(h2('How the estimate improved'));
c.push(table([700,4460,1600,1600],[
 ['Step','What changed','Elevation','Error'],
 ['2.1','First fit: every object in the area casts shadows','57.0°','+1.5°'],
 ['2.2','Only the chair casts shadows (grass blades and noise removed)','55.0°','-0.5°, but unstable'],
 ['2.3','Also score the shadow the chair casts on itself','Did not work alone; set aside','-'],
 ['2.4','Compare shadow outlines instead of shadow areas','56.3°','+0.8°, stable'],
 ['2.5','Test whether the ground is at the right height','Shadow lies on the grass tips','-'],
 ['2.6','Fit the sun and the grass-tip height together','55.08°','-0.43°'],
]));
c.push(sp());
c.push(p('**Reported result: 55° ± 1°** (one scene so far).'));
c.push(p('**Why step 2.3 failed:** the backrest slats are only 2 to 3 cm thick, and the test could not tell a slat\'s front face from its back face, so back faces picked up "lit" votes from the front.'));
c.push(p('**Not done yet:** the brief also suggests using shading gradients (how brightness changes with the angle of a surface) as a second clue for the sun.'));
c.push(p('In Figures 2.1 to 2.5 the panels are: the shadow seen in the photos, the predicted shadow, and where they agree (green = both, blue = predicted only, orange = seen only).'));
c.push(...fig('s2_01_sun_fit_all_casters.jpg','Figure 2.1','Step 2.1, everything casts shadows. Grass blades and stray blobs cast false shadows across the lawn (blue). Elevation 57.0°, overlap score (IoU) 0.14.'));
c.push(...fig('s2_02_sun_fit_chair_only.jpg','Figure 2.2','Step 2.2, only the chair casts shadows. The predicted shadow, including the sunlit gap under the seat, matches closely. Elevation 55.0°, IoU 0.69.'));
c.push(...fig('s2_03_sun_self_shadow.jpg','Figure 2.3','Step 2.3, adding the shadow the chair casts on itself. On its own this did not find the sun, because the thin slats confuse the votes.'));
c.push(...fig('s2_04_sun_boundary_fit.jpg','Figure 2.4','Step 2.4, comparing shadow outlines (orange = seen, blue = predicted, green = both) and the score for each elevation. The outlines agree to about 1.5 cm. Elevation 56.3°.'));
c.push(...fig('s2_05_sun_boundary_fitted_grass_offset.jpg','Figure 2.5','Step 2.6, final fit with the grass-tip height fitted at the same time. Elevation 55.08° vs the true 55.51°.'));

// Stage 3
c.push(h1('5. Stage 3: Grouping blobs into surfaces, with a grain direction'));
c.push(p('**What:** split the chair\'s blobs into continuous surfaces (seat, backrest slats, arms, legs), and find the texture direction on each one.'));
c.push(p('**Why:** stage 5 must only borrow colour from the **same** surface, and along its grain, so that texture lines continue instead of smearing.'));
c.push(p('**How:** each blob is a flattened ellipsoid, so its shortest axis points the way the surface faces (the surface **normal**). Neighbouring blobs join a group as long as their normals stay within 30° of the group\'s average. This is called **region growing**, and it works like a flood fill. Colour is not used, so a shadow edge cannot split a surface in two. The grain direction comes from the direction each blob is stretched in, because training stretches blobs along visible texture.'));
c.push(p('**Problems solved:** the normals of real splat blobs are noisy (24° to 45° off). The first version chained the whole chair into a single group. This was fixed by comparing each blob with the group\'s running average and only trusting normals that two independent methods agree on.'));
c.push(p('**Result:** the chair splits into its real parts. Checked against the photos, the grain matches the photo texture direction to a median of **9°**, where a random direction gives 44°.'));
c.push(...fig('s3_01_chair_clusters.jpg','Figure 3.1','The chair split into surfaces, one colour per surface (grey = not grouped). Top-down view and two side views.'));
c.push(...fig('s3_02_grain_vs_photo_interior.jpg','Figure 3.2','Grain direction compared with the real photo on flat surfaces (green = within 20°, orange = 20° to 45°, red = more than 45° off). Median 9.1°. The red marks gather along the shadow edge.',520));
c.push(...fig('s3_03_grain_vs_photo_edges.jpg','Figure 3.3','The same check on edges and outlines. Median 5.5°. Outlines partly agree by construction, so Figure 3.2 is the stronger test.',520));

// Stage 4
c.push(h1('6. Stage 4: Finding which blobs are in shadow'));
c.push(p('Stage 4 uses several methods that do not depend on each other, so that each one can check the others.'));
c.push(h2('4.1 The sun-blocked test (from the brief)'));
c.push(p('For each blob, look toward the sun: is any other part of the chair in the way? This works like a **shadow map** in computer graphics. **[Addition]** A surface turned away from the sun is also dark, even when nothing blocks it (an **attached shadow**). The brief\'s blocked test misses this, so a facing test was added. It needs to know which side of each surface faces outward, which comes from the cameras that saw it.'));
c.push(p('**Result:** the predicted shadow on the grass matches the photos (98% of shadowed ground caught). On the chair it is right on the legs, under the seat and behind the backrest, but it over-predicts shadow on the backrest front, where the sun hits almost edge-on.'));
c.push(...fig('s4_01_geometry_front.jpg','Figure 4.1','Sun-geometry prediction, front view: photo, prediction (red = shadow, cyan = lit), and the SDDNet shadow outline (magenta).'));
c.push(...fig('s4_02_geometry_rear.jpg','Figure 4.2','Rear view. The back of the backrest is dark in the photo, predicted as shadow, and SDDNet agrees.'));
c.push(...fig('s4_03_geometry_side_views.jpg','Figure 4.3','Chair side views: shadow seen in the photos, prediction, facing away from the sun, and shadow cast by other parts. Legs and the area under the seat agree; the backrest front is over-predicted.'));
c.push(h2('4.2 [Addition] Shadow from photo brightness'));
c.push(p('Each chair blob is looked up in every photo that sees it (about 128 photos per blob) and given its typical brightness. Shaded plastic turns out to be about **13 times darker** than sunlit plastic, the same sun-to-sky ratio measured on the grass, so the blobs fall into two clear groups. **Result:** 45% of the chair is in shadow. This method shares nothing with 4.1, and the two agree well.'));
c.push(...fig('s4_04_brightness_histogram.jpg','Figure 4.4','Chair brightness against sun angle (left) and a histogram of brightness (right). There are two clear groups: sunlit and shaded.',520));
c.push(...fig('s4_05_brightness_front.jpg','Figure 4.5','Shadow from photo brightness, front view: photo, detected (red = shadow, cyan = lit), SDDNet outline. The dark arc on the upper backrest is found; SDDNet outlines only part of it.'));
c.push(...fig('s4_06_brightness_threequarter.jpg','Figure 4.6','Shadow from photo brightness, three-quarter view: shaded upper backrest, lit middle band, lit arm tops, shaded inner legs.'));
c.push(h2('4.3 [Addition] Combining the two'));
c.push(p('Brightness decides, except where the sun hits almost edge-on (16% of the chair blobs). There, weak sun and shade look alike, so the geometry from 4.1 decides. **Result:** 46% of the chair in shadow.'));
c.push(h2('4.4 [Addition] Retraining the shadow detector'));
c.push(p('SDDNet missed most of the chair\'s shadow: it called only 18% of the chair shadowed, because the dark navy plastic fools it. It was retrained for about 2.5 minutes using the 3D labels from 4.3, projected back into the photos, so no hand labelling was needed. It was then tested on frames kept out of training.'));
c.push(p('**Result:** on those frames, its overlap with the labels rose from 0.38 to 0.64, and its overlap with the independent sun geometry rose from 0.21 to 0.48, so it learned something physically consistent. **Caveat:** this is one scene, so it shows the detector can learn, not that it works everywhere.'));
c.push(...fig('s4_10_training_target.jpg','Figure 4.7','The automatic training target: photo, and the target (red = shadow, yellow outline = ignored). Off the chair it uses SDDNet\'s own mask; on the chair it uses the 3D labels.'));
c.push(...fig('s4_11_retrained_detector_649.jpg','Figure 4.8','Frame 649, never seen in training: photo, original SDDNet, retrained SDDNet, 3D labels (red = shadow). The retrained detector marks the whole backrest arc and the inner sides.'));
c.push(...fig('s4_12_retrained_detector_601.jpg','Figure 4.9','Frame 601, never seen in training, in the same layout.'));
c.push(h2('4.5 [Addition] Adjacent brightness-and-colour test'));
c.push(p('A blob is called shadowed when it is much darker (under 0.3 times) than the sunlit blobs around it, within about 30 cm on the same object, **and** its colour still matches theirs. The colour check stops dark paint being mistaken for shadow. A finding along the way: shaded plastic is measurably bluer, because only the blue sky lights it, which is a real shadow signature, and the colour check allows for it.'));
c.push(p('**Result:** 44.5% of the chair is in shadow, consistent with 4.2. This test should work better on objects with several colours, where a single brightness split would fail. It is the labelling used for stage 5.'));
c.push(...fig('s4_07_user_adjacent_method_front.jpg','Figure 4.10','Adjacent brightness-and-colour test, front view: photo, detected (red = shadow, cyan = lit), SDDNet outline.'));
c.push(...fig('s4_08_user_adjacent_method_threequarter.jpg','Figure 4.11','Adjacent brightness-and-colour test, three-quarter view.'));
c.push(...fig('s4_09_user_adjacent_method_rear.jpg','Figure 4.12','Adjacent brightness-and-colour test, rear view. Some lit front-face points show through the thin slats; this is a display effect.'));

// Stage 5
c.push(h1('7. Stage 5: Repainting the shadows'));
c.push(p('**What:** give each shadowed blob the colour it would have in sunlight, taken from sunlit blobs on the same surface.'));
c.push(p('**The rule from the brief:** borrow only from blobs that are on the **same surface** (the stage 3 group) **and lit**, preferring ones along the grain. Not simply the brightest blobs nearby, which could come from another material or a highlight.'));
c.push(h2('How the first test works'));
c.push(bl('For each of the 11,287 shadowed chair blobs (out of 24,445), find the sunlit blobs on the same surface. Steps across the grain count as 4 times longer than steps along it, so blobs along the grain are preferred. The nearest 8 are used.'));
c.push(bl('Very dim "lit" blobs, at the edge of a shadow or barely lit, are not used as sources. Using them left the first try only half as bright as the sunlit parts.'));
c.push(bl('Two ways of applying the colour were tested. **Copy:** the blob takes the sources\' colour outright; brightest, but it flattens the texture. **Ratio:** the blob keeps its own texture and is brightened by the ratio between the sunlit sources and its shadowed neighbours; more natural looking.'));
c.push(bl('Two small surfaces under and inside the chair have almost no sunlit blobs (12.5% of the shadowed blobs). With nothing on the same surface to borrow from, they borrow from the nearest sunlit blobs on any chair part. This breaks the same-surface rule and is noted as a limit.'));
c.push(h2('First results'));
c.push(p('Median brightness of the shadowed blobs went from **0.026** before to **0.229** (copy) or **0.226** (ratio) after, compared with **0.262** for the sunlit parts. The shadows are lifted to about **87%** of sunlit brightness. Each fill takes about 2 seconds. The original splat is not changed; the results are new files.'));
c.push(p('In renders from the real camera positions, the dark arc across the backrest and the darker inner legs are mostly gone, including on a frame not used for any tuning. The copy version looks washed out and too light blue in places; the ratio version keeps more of the plastic texture.'));
c.push(new Paragraph({spacing:{after:120},children:[new TextRun({text:'[Before and after images for stage 5 to be added here: photo, original, copy, ratio.]',italics:true,color:'888888'})]}));
c.push(h2('Known problems to fix next'));
c.push(bl('**Mottling:** the shadow labels are speckled, so neighbouring blobs get filled differently. The labels need smoothing in 3D, and the fill needs blending at shadow edges.'));
c.push(bl('A slight brightness and colour mismatch with the real sunlit plastic.'));
c.push(bl('Thin slats: their front and back faces interfere with each other.'));
c.push(bl('The chair\'s shadow on the grass is not filled yet; this first test covered the chair only.'));

// Limits
c.push(h1('8. Limits and next steps'));
c.push(bl('Everything so far is tested on **one scene**. More captures are needed to show the method holds in general.'));
c.push(bl('The sun\'s compass direction cannot be checked, because the scene has no built-in north. Only the elevation is checked.'));
c.push(bl('The self-shadow version of the sun fit (step 2.3) is set aside until thin surfaces are handled.'));
c.push(bl('The shading-gradient clue for the sun, suggested in the brief, is not tried yet.'));
c.push(bl('The sun-blocked test over-predicts shadow where the sun hits almost edge-on.'));
c.push(bl('The retrained detector was trained and tested on the same scene, on separate frames.'));
c.push(bl('The method is still being tested and is not yet part of the standard splat process.'));

// Glossary
c.push(h1('9. Glossary'));
[['Gaussian, splat','One soft coloured blob; a whole scene is millions of them.'],
 ['Structure from Motion (SfM), COLMAP','A method and program that recover camera positions and a rough 3D point cloud from photos.'],
 ['Normal','The direction a surface faces.'],
 ['Grain','The direction of a surface\'s texture, such as wood lines or moulded ridges.'],
 ['Cast shadow','Shadow caused by something blocking the sun.'],
 ['Attached shadow','Darkness on a surface turned away from the sun.'],
 ['Shadow map','A graphics technique that checks, for each point, whether anything lies between it and the light.'],
 ['Region growing','Grouping neighbouring points step by step while they stay similar, like a flood fill.'],
 ['SDDNet','An existing neural network that marks shadow pixels in a photo.'],
 ['IoU (intersection over union)','An overlap score between two regions: 1 means identical, 0 means no overlap.'],
 ['Held-out frames','Photos kept out of training and used only for testing.'],
].forEach(([a,b])=>c.push(new Paragraph({spacing:{after:100},children:[new TextRun({text:a+': ',bold:true}),new TextRun(b)]})));

const doc=new Document({
 creator:'Claude',title:'Removing baked-in sun shadows from a Gaussian splat',
 styles:{default:{document:{run:{font:'Calibri',size:22}}},
  paragraphStyles:[{id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,run:{size:32,bold:true,color:'1F3A5F'},paragraph:{spacing:{before:240,after:200},outlineLevel:0}},
   {id:'Heading2',name:'Heading 2',basedOn:'Normal',next:'Normal',quickFormat:true,run:{size:26,bold:true,color:'2E5C8A'},paragraph:{spacing:{before:240,after:120},keepNext:true,outlineLevel:1}}]},
 numbering:{config:[{reference:'b',levels:[{level:0,format:LevelFormat.BULLET,text:'•',alignment:AlignmentType.LEFT,style:{paragraph:{indent:{left:540,hanging:270}}}}]}]},
 sections:[{properties:{page:{size:{width:12240,height:15840},margin:{top:1440,bottom:1440,left:1440,right:1440}}},
  footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({children:[PageNumber.CURRENT],size:18,color:'888888'})]})]})},
  children:c}]});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync(OUT,b);console.log('wrote',OUT,b.length)});
