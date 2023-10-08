from CF_personalised import rmse_evaluations as cf_rmse_evaluations
from CF_personalised import nDCG_hitRate_evaluations as cf_nDCG_hitRate_evaluations
from non_personalised import non_personalised_rmse_nDCG_hitRate_evaluations as non_personalised_rmse_nDCG_hitRate_evaluations

print("Evaluating Collaborative Filtering...")
print("Evaluating rmse...")
cf_rmse = cf_rmse_evaluations()
print("Evaluating nDCG and hitRate when k=10...")
cf_nDCG_10,cf_hitRate_10 = cf_nDCG_hitRate_evaluations(k=10)
print("Evaluating nDCG and hitRate when k=100...")
cf_nDCG_100,cf_hitRate_100 = cf_nDCG_hitRate_evaluations(k=100)
print("Evaluating nDCG and hitRate when k=5...")
cf_nDCG_5,cf_hitRate_5 = cf_nDCG_hitRate_evaluations(k=5)
print("Evaluating Non-Personalised when k=5...")
np_rmse,np_nDCG_10,np_hitRate_5 = non_personalised_rmse_nDCG_hitRate_evaluations(k=5)
print("Evaluating Non-Personalised when k=10...")
np_rmse,np_nDCG_10,np_hitRate_10 = non_personalised_rmse_nDCG_hitRate_evaluations(k=10)
print("Evaluating Non-Personalised when k=100...")
np_rmse,np_nDCG_100,np_hitRate_100 = non_personalised_rmse_nDCG_hitRate_evaluations(k=100)

print("RMSE: Collaborative Filtering: ",round(cf_rmse,2)," Non-Personalised: ",round(np_rmse,2))
print("nDCG10: Collaborative Filtering: ",round(cf_nDCG_10,2)," Non-Personalised: ",round(np_nDCG_10,2))
print("nDCG100: Collaborative Filtering: ",round(cf_nDCG_100,2)," Non-Personalised: ",round(np_nDCG_100,2))
print("HitRate(k=5): Collaborative Filtering: ",round(cf_hitRate_5,2)," Non-Personalised: ",round(np_hitRate_5,2))
print("HitRate(k=10): Collaborative Filtering: ",round(cf_hitRate_10,2)," Non-Personalised: ",round(np_hitRate_10,2))