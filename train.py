from pathlib import Path
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
P=Path(__file__).parent
d=pd.read_csv(P/'data.csv')
features=[c for c in d.columns if c not in ['tactic_label','match_id','phase_index','match_minute','period','duration','progress_speed_mps']]+['match_minute','period','duration','progress_speed_mps']
assert len(features)==18 and not d[features].isna().any().any()
assert not d.duplicated(['match_id','phase_index']).any()
train=d[~d.match_id.isin([2015213,1899585])]; test=d[d.match_id.isin([2015213,1899585])]
enc=LabelEncoder().fit(d.tactic_label); y=enc.transform(train.tactic_label)
m=XGBClassifier(objective='multi:softprob',num_class=len(enc.classes_),n_estimators=400,max_depth=5,learning_rate=.05,subsample=.8,colsample_bytree=.8,eval_metric='mlogloss',tree_method='hist',random_state=42,n_jobs=4)
m.fit(train[features],y,sample_weight=compute_sample_weight('balanced',y))
pred=enc.inverse_transform(m.predict(test[features]).astype(int))
m.save_model(P/'model.ubj')
meta=dict(features=features,classes=enc.classes_.tolist(),accuracy=accuracy_score(test.tactic_label,pred),macro_f1=f1_score(test.tactic_label,pred,average='macro'),train_phases=len(train),test_phases=len(test),test_matches=[2015213,1899585],confusion=confusion_matrix(test.tactic_label,pred,labels=enc.classes_).tolist())
(P/'metadata.json').write_text(json.dumps(meta,indent=2));test.to_csv(P/'examples.csv',index=False)
print(json.dumps(meta,indent=2))
