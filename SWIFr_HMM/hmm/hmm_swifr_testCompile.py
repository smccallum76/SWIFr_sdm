from sqlalchemy import create_engine
import hmm_funcs as hmm

update_db = input("Would you like to update the DB, 'yes' or 'no': ")
# update_db = 'no'
db_name = 'hmm_predictions.db'
swifr_table_name = 'swifr_pred'  # stochastic backtrace (sbt)
db_path = 'C:/Users/scott/PycharmProjects/SWIFr_sdm/SWIFr_HMM/hmm/output_db/'

if update_db == 'yes':
    engine = create_engine('sqlite:///' + db_path + db_name, echo=False)

gmm_path = '../../swifr_pkg/test_data/thesis_4class/'
swifr_fst_path = '../../swifr_pkg/test_data/thesis_4class_1stat/class4_fst_test/test/test_classified'
swifr_xpehh_path = '../../swifr_pkg/test_data/thesis_4class_1stat/class4_xpehh_test/test/test_classified'
swifr_ihs_path = '../../swifr_pkg/test_data/thesis_4class_1stat/class4_ihs_test/test/test_classified'
swifr_allStat_path = '../../swifr_pkg/test_data/thesis_4class_test/test/test_classified'

swifr_fst = hmm.hmm_get_data(swifr_fst_path).reset_index(drop=False)
swifr_fst = swifr_fst.rename(columns={'index': 'idx_key'})

swifr_xpehh = hmm.hmm_get_data(swifr_xpehh_path).reset_index(drop=False)
swifr_xpehh = swifr_xpehh.rename(columns={'index': 'idx_key'})

swifr_ihs = hmm.hmm_get_data(swifr_ihs_path).reset_index(drop=False)
swifr_ihs = swifr_ihs.rename(columns={'index': 'idx_key'})

swifr_all = hmm.hmm_get_data(swifr_allStat_path).reset_index(drop=False)
swifr_all = swifr_all.rename(columns={'index': 'idx_key'})

# save the data
if update_db == 'yes':
    swifr_fst.to_sql(f"{swifr_table_name}_fst_4class", con=engine, if_exists='replace')
    swifr_xpehh.to_sql(f"{swifr_table_name}_xpehh_4class", con=engine, if_exists='replace')
    swifr_ihs.to_sql(f"{swifr_table_name}_ihs_4class", con=engine, if_exists='replace')
    swifr_all.to_sql(f"{swifr_table_name}_allStats_4class", con=engine, if_exists='replace')

print('done')