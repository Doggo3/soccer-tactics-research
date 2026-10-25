from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from definitions import FIELDS
from paper_web import render as render_paper

P=Path(__file__).parent
NAMES={'build_up':'Build-up','creation':'Creation','direct_attack':'Direct attack','fast_transition':'Fast transition','finishing':'Finishing'}
COLORS=dict(zip(NAMES.values(),['#28736b','#447eb0','#a76a26','#8c5aa1','#bc5554']))
DEFS={'build_up':'A controlled attack, normally beginning farther from the opponent’s goal.','creation':'Developing an attacking opportunity.','direct_attack':'Moving the ball forward quickly.','fast_transition':'Attacking soon after gaining possession.','finishing':'The final part of an attack near the opponent’s goal.'}
st.set_page_config(page_title='Classifying Soccer Attacking Phases',layout='wide')
st.markdown('''<style>
.block-container{max-width:1160px;padding-top:3rem;padding-bottom:4rem}h1{font-size:2.2rem!important;line-height:1.15!important;letter-spacing:-.045em}h2{font-size:1.6rem!important;letter-spacing:-.025em}h3{font-size:1.15rem!important}p,li{line-height:1.6} [data-testid="stMetric"]{background:#edf3ef;border:1px solid #d9e4dc;border-radius:10px;padding:14px} [data-testid="stMetricValue"]{font-size:1.65rem} [data-testid="stCaptionContainer"] p{color:#4f6057}button{min-height:44px} .step{border-top:3px solid #397563;padding:15px 0 8px}.step strong{font-size:18px}.step p{color:#4f6057}div[data-testid="stRadio"]>div{gap:.5rem 1.1rem}hr{margin:1.4rem 0}
</style>''',unsafe_allow_html=True)
@st.cache_resource
def load():
 meta=json.loads((P/'metadata.json').read_text());model=XGBClassifier();model.load_model(P/'model.ubj')
 full=pd.read_csv(P/'data.csv');full['Split']=np.where(full.match_id.isin(meta['test_matches']),'Test: 2 unseen matches','Train: 8 matches')
 test=full[full.match_id.isin(meta['test_matches'])].reset_index(drop=True).copy()
 probs=model.predict_proba(test[meta['features']]);test['prediction']=np.array(meta['classes'])[probs.argmax(axis=1)];test['correct']=test.tactic_label==test.prediction
 return model,meta,full,test,probs,json.loads((P/'paper_results.json').read_text())
m,meta,full,test,probs,paper=load();features=meta['features'];classes=meta['classes']

def chart(fig,key,height=370):
 fig.update_layout(template='plotly_white',height=height,font=dict(family='Arial, sans-serif',size=13,color='#243e31'),margin=dict(l=12,r=15,t=36,b=40),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',legend=dict(title=None,orientation='h',y=-.22,x=0),hoverlabel=dict(font_size=13))
 fig.update_xaxes(automargin=True);fig.update_yaxes(automargin=True)
 st.plotly_chart(fig,width='stretch',key=key,config={'displaylogo':False,'scrollZoom':False,'modeBarButtonsToRemove':['lasso2d','select2d']})
def phase_name(row):
 sec=int(round(row.match_minute*60));return f'Match {int(row.match_id)} · phase {int(row.phase_index)} · half {int(row.period)}, {sec//60}:{sec%60:02d}'
def measurements(row):
 return pd.DataFrame([{'Measurement':FIELDS[f][0],'Value':float(row[f]),'Unit':FIELDS[f][1],'Meaning':FIELDS[f][2]} for f in features])
def goto(page):st.session_state['page']=page

def scope_note():
 st.caption('Original study: supplied paper and saved notebook output. Live predictions: a rerun of the notebook’s Enhanced XGBoost configuration on the same CSV and match split.')

st.title('Classifying soccer attacking phases')
st.caption('A study of five attacking phases using player and ball tracking measurements')
SECTIONS={'study':'The study','classify':'Classify a phase','data':'Explore the data','results':'Results & method','paper':'Paper'}
if 'page' not in st.session_state:st.session_state.page=SECTIONS.get(st.query_params.get('section','study'),'The study')
page=st.radio('Section',list(SECTIONS.values()),horizontal=True,key='page',label_visibility='collapsed')
slug=next(k for k,v in SECTIONS.items() if v==page)
if st.query_params.get('section')!=slug:st.query_params['section']=slug
st.divider()

if page=='The study':
 st.header('Can measurements tell us what kind of attack is happening?')
 st.write('One section of play, 18 measurements, five possible labels. This study tests whether a machine-learning model can identify an attacking phase from the positions and movement of the players and ball.')
 a,b,c=st.columns(3)
 a.markdown('<div class="step"><strong>1. Measure a phase</strong><p>Summarize one section of play using 18 inputs: position, team shape, pressure, and timing.</p></div>',unsafe_allow_html=True)
 b.markdown('<div class="step"><strong>2. Predict its label</strong><p>XGBoost scores the five possible categories and selects the highest-scoring one.</p></div>',unsafe_allow_html=True)
 c.markdown('<div class="step"><strong>3. Check the answer</strong><p>Compare the prediction with SkillCorner’s existing label for that recorded phase.</p></div>',unsafe_allow_html=True)
 a,b=st.columns(2)
 a.button('Try classifying a real phase',type='primary',on_click=goto,args=('Classify a phase',))
 b.button('Read the paper',on_click=goto,args=('Paper',))
 st.subheader('Five labels, five different meanings')
 for c in classes:st.markdown('**'+NAMES[c]+'** — '+DEFS[c])
 st.write('**A key distinction:** direct attack describes quick forward movement; fast transition also needs the context of a recent change of possession. These labels are categories, not five steps every attack must follow.')
 st.subheader('The evidence behind the study')
 a,b,c=st.columns(3);a.metric('Recorded phases',f'{len(full):,}');b.metric('A-League matches',str(full.match_id.nunique()));c.metric('Held-out test phases',str(len(test)))
 st.write('Eight matches (2,388 phases) were used for training. Two different matches (567 phases) were reserved for testing. Keeping entire matches separate tests whether the model can work on games it did not train on.')
 st.info('The paper reports 91.9% accuracy for Enhanced XGBoost. But it correctly identified only 13 of 25 fast transitions. Strong overall accuracy can hide a weaker result for a less common category.')
 st.caption('Source: paper, Dataset, Methodology, and Results; notebook cells 3–7. The reference labels come from SkillCorner’s system, not a separate assessment by coaches in this study.')

elif page=='Classify a phase':
 st.header('Give the classifier a real recorded phase')
 st.write('Choose an example from the two held-out test matches, inspect its measurements, then run the classifier. Each example uses its complete set of 18 recorded inputs.')
 a,b=st.columns(2)
 label=a.selectbox('Reference label to explore',['All five labels']+classes,format_func=lambda c:NAMES.get(c,c))
 outcome=b.selectbox('Examples to show',['All examples','Model matches reference','Model differs from reference'])
 pool=test if label=='All five labels' else test[test.tactic_label==label]
 if outcome!='All examples':pool=pool[pool.correct==(outcome=='Model matches reference')]
 if pool.empty:
  st.info('No examples match these filters. Choose “All examples” or another label.')
 else:
  options=pool.index.tolist();key=f'phase_{label}_{outcome}'
  typical=(pool[features].rank(pct=True)-.5).abs().mean(axis=1)
  if st.session_state.get(key) not in options:st.session_state[key]=int(typical.idxmin())
  idx=st.selectbox('Recorded test phase',options,format_func=lambda i:phase_name(test.loc[i]),key=key)
  position=options.index(idx)
  def next_record():
   st.session_state[key]=options[position+1]
   st.session_state.pop('classified_id',None)
  st.button('Next recorded phase',disabled=position==len(options)-1,on_click=next_record)
  row=test.loc[idx];st.caption(f'{len(pool)} available examples. The clock is elapsed time within the half, restarting at zero in half 2.')
  st.subheader('What the model receives')
  a,b,c=st.columns(3);a.metric('Phase duration',f'{row.duration:.1f} s');b.metric('Average attacking width',f'{row.attack_width_mean:.1f} m');c.metric('Defenders within 5 m',f'{row.defenders_within_5m_mean:.2f} on average')
  st.write('Width describes the team’s side-to-side spread. Nearby-defender counts describe pressure around the ball. These are averages over the phase, so a player count can be fractional.')
  with st.expander('Inspect all 18 model inputs'):
   st.dataframe(measurements(row),hide_index=True,width='stretch',column_config={'Value':st.column_config.NumberColumn(format='%.3f')})
   st.caption('Match ID, phase ID, and reference label are identifiers or evaluation data. They are not model inputs. Signed x movement is not automatically forward or backward relative to the attacking team.')
  if st.button('Classify this phase',type='primary'):st.session_state['classified_id']=int(idx)
  if st.session_state.get('classified_id')==idx:
   current=m.predict_proba(pd.DataFrame([row[features].astype(float)],columns=features))[0];winner=classes[int(current.argmax())]
   st.subheader('What the classifier returned')
   a,b=st.columns(2);a.metric('Predicted label',NAMES[winner]);b.metric('Dataset reference label',NAMES[row.tactic_label])
   if winner==row.tactic_label:st.success('The predicted label matches the dataset’s reference label.')
   else:st.warning('The predicted label differs from the dataset’s reference label. This counts as an error in the test evaluation.')
   st.write('**Reference meaning:** '+DEFS[row.tactic_label])
   if winner!=row.tactic_label:st.write('**Predicted meaning:** '+DEFS[winner])
   scores=pd.DataFrame({'Label':[NAMES[c] for c in classes],'Score':current}).sort_values('Score')
   fig=px.bar(scores,x='Score',y='Label',orientation='h',text='Score',range_x=[0,1],color='Label',color_discrete_map=COLORS)
   fig.update_traces(texttemplate='%{x:.1%}');fig.update_xaxes(tickformat='.0%',title='Model probability');fig.update_layout(showlegend=False);chart(fig,'phase_scores',310)
   st.caption('The highest score determines the prediction. These probabilities are not guarantees, and the chart does not explain which input caused this particular decision.')
   st.write('**What to try next:** select a different recorded phase, or choose “Model differs from reference” to inspect an error. The new record brings its own complete set of 18 inputs.')
   csv=measurements(row).to_csv(index=False).encode()
   st.download_button('Download this phase’s measurements',csv,file_name=f'phase_{int(row.match_id)}_{int(row.phase_index)}.csv',mime='text/csv')
  st.divider()
  st.caption(f'Live model: rerun of notebook cells 3–4 (400 trees, depth 5, learning rate 0.05). It scores {meta["accuracy"]:.1%} accuracy and {meta["macro_f1"]:.3f} macro F1 here. The notebook’s saved run reports 91.9% and 0.864; its original fitted model was not included, so this is not presented as that exact saved run.')

elif page=='Explore the data':
 st.header('Look for patterns in the recorded measurements')
 st.write('Every point is one actual phase. Compare categories and see where their measurements overlap. A pattern in this view does not establish a rule for assigning a tactic.')
 a,b=st.columns([1,2]);split=a.selectbox('Data split',['All 10 matches','Train: 8 matches','Test: 2 unseen matches'])
 labels=b.multiselect('Included labels',classes,default=classes,format_func=lambda c:NAMES[c])
 view=full if split=='All 10 matches' else full[full.Split==split]
 view=view[view.tactic_label.isin(labels)].copy();view['Label']=view.tactic_label.map(NAMES)
 if view.empty:st.info('No labels selected. Select at least one label to show its records.')
 else:
  st.caption(f'{len(view):,} phases · {view.match_id.nunique()} matches · all charts and downloads below use these filters')
  options=['attack_width_mean','duration','ball_x_start','ball_progress_m','attack_depth_mean','defenders_within_5m_mean','attackers_ahead_of_ball_mean']
  a,b=st.columns(2);x=a.selectbox('Horizontal measurement',options,format_func=lambda f:FIELDS[f][0]);y=b.selectbox('Vertical measurement',options,index=1,format_func=lambda f:FIELDS[f][0])
  st.caption(FIELDS[x][2]+' '+FIELDS[y][2])
  fig=px.scatter(view,x=x,y=y,color='Label',color_discrete_map=COLORS,category_orders={'Label':list(NAMES.values())},hover_data={'match_id':True,'phase_index':True,'period':True,'Label':True},labels={x:f'{FIELDS[x][0]} ({FIELDS[x][1]})',y:f'{FIELDS[y][0]} ({FIELDS[y][1]})','match_id':'Match','phase_index':'Phase','period':'Half'})
  fig.update_traces(marker=dict(size=6,opacity=.6));chart(fig,'data_scatter',430)
  st.caption('Hover to inspect a record. Click a legend label to hide that category in this chart; use the filters above to change the actual data selection.')
  st.subheader('How much the categories overlap')
  f=st.selectbox('Compare the distribution of',options,format_func=lambda f:FIELDS[f][0],key='distribution_field')
  fig=px.box(view,x=f,y='Label',orientation='h',color='Label',color_discrete_map=COLORS,category_orders={'Label':list(NAMES.values())},labels={f:f'{FIELDS[f][0]} ({FIELDS[f][1]})'})
  fig.update_layout(showlegend=False);chart(fig,'data_boxes',330)
  st.caption('The line inside a box is the median. The box holds the middle 50% of values; points beyond the whiskers are unusually far from the middle. Overlap means this measurement alone cannot cleanly separate those labels.')
  counts=view.tactic_label.value_counts().reindex(classes,fill_value=0)
  st.subheader('How many examples are available?')
  frame=pd.DataFrame({'Label':[NAMES[c] for c in classes if c in labels],'Phases':[int(counts[c]) for c in classes if c in labels]})
  fig=px.bar(frame,x='Phases',y='Label',orientation='h',text='Phases',color='Label',color_discrete_map=COLORS);fig.update_layout(showlegend=False);chart(fig,'data_counts',280)
  with st.expander('Inspect and download the filtered records'):
   st.dataframe(view.drop(columns=['Label']).head(100),hide_index=True,width='stretch');st.caption('Preview: first 100 filtered records. The download includes every filtered record.')
   st.download_button('Download filtered data',view.drop(columns=['Label']).to_csv(index=False).encode(),file_name='soccer_phases_filtered.csv',mime='text/csv')
  with st.expander('Data definitions and checks'):
   st.write('The CSV has 2,955 rows, 10 matches, no missing values, and no duplicate match/phase identifiers. Each row summarizes a labeled phase; there are no player paths or videos in this file.')
   st.write('Some stored progress-speed values are extreme (−252.5 to +190.0 m/s). Signed progress divided by duration reproduces them, but the original extraction code is not supplied, so they should not be interpreted as physical ball speed. The count of attackers ahead also exceeds 10 in one record; its counting convention needs verification.')
   st.write('The notebook trains on existing summary columns; it does not define the underlying spacing, wide-zone, or attacking-direction extraction. The app preserves those columns without inventing definitions or reconstructed positions.')


elif page=='Paper':
 render_paper(paper,chart,goto)

else:
 st.header('What the research found')
 st.write('These results come from the paper and the notebook’s saved output. They describe classification of existing SkillCorner labels, not predictions of player movement or whether an attack will succeed.')
 tab1,tab2,tab3=st.tabs(['Classification results','Inputs & validation','Method & sources'])
 with tab1:
  a,b,c=st.columns(3);a.metric('Enhanced XGBoost accuracy','91.9%');b.metric('Macro F1','0.864');c.metric('Correct test labels','521 / 567')
  st.caption('Accuracy is the fraction of all test labels predicted correctly. Precision asks how often a predicted category is right; recall asks how many actual examples of that category were found. F1 combines precision and recall. Macro F1 averages those five F1 scores equally.')
  st.subheader('Which labels were confused?')
  chosen=st.selectbox('Inspect a reference label',classes,index=3,format_func=lambda c:NAMES[c])
  ci=classes.index(chosen);cm=np.asarray(paper['confusion']);n=int(cm[ci].sum());correct=int(cm[ci,ci])
  st.write(f'Of **{n} {NAMES[chosen].lower()} phases**, **{correct} were correctly identified** ({correct/n:.1%}). The other {n-correct} received a different label.')
  frame=pd.DataFrame({'Predicted label':[NAMES[c] for c in classes],'Phases':cm[ci]})
  fig=px.bar(frame,x='Phases',y='Predicted label',orientation='h',text='Phases',color='Predicted label',color_discrete_map=COLORS,range_x=[0,n]);fig.update_layout(showlegend=False);chart(fig,'confusion_row',290)
  if chosen=='fast_transition':st.write('Five were labeled creation and six finishing; one was labeled direct attack. The paper argues that phase averages lose the sequence of events that helps distinguish a transition. This is a proposed explanation, not proof of the cause of each error.')
  with st.expander('See the complete confusion matrix'):
   chart(px.imshow(cm,x=list(NAMES.values()),y=list(NAMES.values()),text_auto=True,color_continuous_scale='Greens',labels={'x':'Predicted label','y':'Reference label','color':'Phases'}),'paper_confusion',450)
   st.caption('Read by row: reference labels at left, predicted labels below. Diagonal cells are correct. Source: notebook cell 6 saved figure; counts sum to 567.')
  st.subheader('How the models compare')
  metric=st.radio('Comparison metric',['Accuracy','Macro F1'],horizontal=True)
  comparison=pd.DataFrame({'Model':['Original Random Forest','Enhanced Random Forest','Original XGBoost','Enhanced XGBoost','SVC','MLP','KNN'],'Accuracy':[.912,.912,.914,.919,.879,.858,.756],'Macro F1':[.863,.873,.870,.864,.825,.790,.649]})
  fig=px.bar(comparison.sort_values(metric),x=metric,y='Model',text=metric,orientation='h',range_x=[0,1],color_discrete_sequence=['#28736b']);fig.update_traces(texttemplate='%{x:.1%}' if metric=='Accuracy' else '%{x:.3f}')
  if metric=='Accuracy':fig.update_xaxes(tickformat='.0%')
  chart(fig,'model_comparison',370)
  st.caption('Original = 14 inputs; enhanced = 18. Enhanced XGBoost leads accuracy in this comparison; Enhanced Random Forest leads macro F1. Source: paper results table and notebook cell 21.')
 with tab2:
  st.subheader('Which inputs did XGBoost use most?')
  top=st.radio('Number of inputs shown',[5,18],horizontal=True,format_func=lambda n:f'Top {n}' if n==5 else 'All 18')
  imp=pd.DataFrame([{'Measurement':FIELDS[f][0],'Importance':v} for f,v in paper['importance'].items()]).sort_values('Importance',ascending=False).head(top).sort_values('Importance')
  fig=px.bar(imp,x='Importance',y='Measurement',orientation='h',text='Importance',color_discrete_sequence=['#28736b']);fig.update_traces(texttemplate='%{x:.3f}');chart(fig,'importance',max(300,top*27))
  st.caption('Saved notebook cell 8: XGBoost feature-importance scores. These summarize how the fitted model uses inputs across its trees; they do not explain an individual prediction or show cause and effect.')
  st.subheader('Does performance vary between matches?')
  folds=pd.DataFrame({'Fold':range(1,6),'Accuracy':paper['fold_accuracy'],'Macro F1':paper['fold_macro_f1']})
  chart(px.line(folds.melt(id_vars='Fold',var_name='Metric',value_name='Score'),x='Fold',y='Score',color='Metric',markers=True,range_y=[0,1],color_discrete_map={'Accuracy':'#28736b','Macro F1':'#8c5aa1'}),'folds',300)
  st.write('The saved notebook reports mean accuracy **0.905** and mean macro F1 **0.836** across five match-group folds. These folds use a **Random Forest with 300 trees**, not the Enhanced XGBoost model above. Every fold keeps matches together.')
  st.caption('Source: notebook cell 16. Fold-level values are shown to the three decimals saved in its output. Differences between folds show why one test split cannot establish performance in every match.')
  with st.expander('Feature removal and tuning'):
   st.dataframe(pd.DataFrame({'Inputs retained':[14,12,11,9],'Accuracy':[.912,.910,.910,.912],'Macro F1':[.863,.858,.861,.867]}),hide_index=True,width='stretch')
   st.write('The feature-removal experiment uses Random Forest and begins with the 14 original inputs. Changes are small; it does not establish that fewer features are reliably better.')
   st.write('The paper reports a tuning search with 25 configurations and five folds, with best validation macro F1 0.884. Notebook cell 18 records these settings as a summary; it does not contain the search execution. Best reported settings include 400 trees, depth 7, and learning rate 0.03. They are not the depth-5 configuration fitted in notebook cell 4.')
 with tab3:
  st.subheader('What was trained and tested?')
  st.write('The notebook fits Enhanced XGBoost to 2,388 phases from eight matches using 18 inputs and balanced sample weights. It tests on 567 phases from matches 2015213 and 1899585. Labels and match/phase identifiers are excluded from the input features.')
  st.write('The four added inputs are minutes elapsed in the half, half number, duration, and signed progress speed. The original 14 describe ball location, team shape, nearby defenders, and sample counts.')
  st.subheader('Why the live prediction scores differ slightly')
  st.write(f'The notebook includes code and saved outputs, but no fitted model file. The site reruns its cell-4 configuration on the supplied CSV: 400 trees, depth 5, learning rate 0.05, 0.8 row/column sampling, and seed 42. This fitted model reaches {meta["accuracy"]:.1%} accuracy and {meta["macro_f1"]:.3f} macro F1. The saved historical run reaches 91.9% and 0.864. The supplied notebook does not pin package versions, so the exact cause of this small difference has not been established.')
  st.subheader('What the study cannot establish')
  st.write('The dataset contains only 10 matches from one league. Its labels were generated by SkillCorner’s system. High agreement with those labels does not establish agreement with independent coaches, performance in other leagues, or the ability to predict future movement. Phase averages also discard the order of events.')
  st.subheader('Sources used on this site')
  st.markdown('- **Paper:** TS8_ Final Research Paper Task Sheet-4.pdf — definitions, methods, reported results, and limitations.\n- **Notebook:** Classifying Soccer Tactics — 22 code cells and saved outputs; model setup, confusion matrix, feature importance, and grouped validation.\n- **Dataset:** skillcorner_enhanced_dataset(1).csv — 2,955 recorded phases and 21 columns, of which 18 are model inputs.')
  st.caption('The original feature-extraction code, match videos, full tracking sequences, and historical fitted model were not included. No invented trajectories or formations are used in this site.')

st.divider()
st.caption('Research scope: classify recorded attacking phases into five categories. Data and reference labels: SkillCorner. Paper acknowledgements: Omar Ramos Escoto and Inspirit AI.')
