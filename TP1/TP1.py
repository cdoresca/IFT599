import pandas as pd
import matplotlib.pyplot as plt
import json 
from sklearn.preprocessing import LabelEncoder
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

user = pd.read_csv('users_data.csv')
card = pd.read_csv('cards_data.csv')
transaction  = pd.read_csv('transactions_data.csv')

with open('mcc_codes.json', 'r') as f:
        mcc_code = json.load(f)



################
#Question 1.1
################

# Fusion des tableaux user et card selon le id user
user_card = pd.merge(user, card, how = 'left', left_on = 'id', right_on='client_id',suffixes=('_user','_card')).drop('client_id',axis = 1)

# Fusion des tableau card et transaction selon le id card
df = pd.merge(user_card, transaction, how = 'left', left_on = 'id_card', right_on='card_id',suffixes=('_user','_transaction')).drop()

print(df)

################
#Question 1.2
#################

# Dans le tableau user, les variables les moins pertinants sont current_age, retirement_age, 
# birth_year, birth_month, latitude, longitude. On n'a pas besoin de savoir l'âge du user si
# on peut le calculer à partir de la date de naissance. Pour birth_year, birth_month, il  est inutile d'avoir 
# 2 colonnes pour la date de naissance. La latitude et la longitude ne sont pas nécessaire lorsqu'on possède l'adresse

# Dans le tableau transaction les variables merchant_city, merchant_state pourrait juste être une adresse.

################
#Question 1.3
#################

# Nettoyage cards
# Ajuster type de données de string à float
card['credit_limit'] = card["credit_limit"].replace(r"[\$,]", "", regex=True).astype(float)

# Ajuster type de données de string à datetime
card["acct_open_date"] = pd.to_datetime(card["acct_open_date"],errors='coerce',format ='%m/%Y')
card["expires"] = pd.to_datetime(card["expires"], errors='coerce',format ='%m/%Y')

# Ajuster type de données de string à bool
card["has_chip"] = card["has_chip"].str.lower().map({"yes": True, "no": False})
card["card_on_dark_web"] = card["card_on_dark_web"].str.lower().map({"yes": True, "no": False})

# Enlever duplicate
card = card.drop_duplicates(subset=["client_id", "card_number"])

# Incohérence
cards_clean = card[card["expires"] > card["acct_open_date"]]

# Nettoyage user
# Ajuster type de données de string à float
for col in ["per_capita_income", "yearly_income", "total_debt"]:
    user[col] =user[col].replace(r"[\$,]", "", regex=True).astype(float)
    

# Enlever duplicate
user = user.drop_duplicates()

# Incohérence
user = user[user["retirement_age"] >= user["current_age"]]
user = user[abs((2025 - user["birth_year"]) - user["current_age"]) <= 2]

# Nettoyage transaction
# Ajuster type de données de string à float
transaction['amount'] =transaction["amount"].replace(r"[\$,]", "", regex=True).astype(float)


# Ajuster type de données de string à datetime
transaction['date'] = pd.to_datetime(transaction["date"], errors='coerce')


################
#Question 1.4
#################

################
#Question 1.5
#################
merchant = transaction.groupby("merchant_id").agg({
    "id": "count",
    "amount": "sum"
}).rename(columns={
    "id": "n_transactions",
    "amount": "total_revenue"
})

top_merchant_revenue = merchant.nlargest(10, "total_revenue")
top_merchant_count   = merchant.nlargest(10, "n_transactions")

top_merchant_revenue.plot(kind = 'pie',y = 'total_revenue')
top_merchant_count.plot(kind = 'pie',y = 'n_transactions')
plt.show()


#################
#Question 1.6
#################

transaction['mcc_category'] = transaction['mcc'].map(mcc_code)

encoder = LabelEncoder()
card['card_type_encode'] = encoder.fit_transform(card["card_type"])

#################
#Question 2
#################
q1 = transaction['amount']. quantile(0.25)
q2 = transaction['amount']. quantile(0.50)
q3 = transaction['amount']. quantile(0.75)

def categorize_amount_quartile ( x ) :
    if x <= q1 :
        return 'amount_low '
    elif x <= q2 :
        return  'amount_medium'
    elif x <= q3 :
        return 'amount_high '
    else :
        return 'amount_very_high'
    
transaction['amount_item '] = transaction['amount'].apply(categorize_amount_quartile)
transaction['credit_score_item '] = transaction['credit_score'].apply( 
     lambda x : 'credit_high' if x >= 700 else ('credit_medium' if
    x >= 500 else 'credit_low')
)
transaction['card_type_item'] = transaction['card_type'].apply( lambda x : f'card_ {x.lower ()}')
transaction['merchant_state_item'] = transaction['merchant_state'].apply( lambda x :f'state_ {x}')
transaction['mcc_item'] = transaction['mcc_description'].apply( lambda x : f'mcc_ {x.replace (" " , "_"). lower ()}')

cols_to_unstack = ['mcc_item ', 'card_type_item ','merchant_state_item ', 'amount_item', 'credit_score_item']
df_base = transaction[[ 'client_id_x ', 'card_id ']]. copy ()
df_base.rename( columns ={ 'client_id_x ': 'client_id '} , inplace = True)
df_long = pd.melt(pd.concat([ df_base , transaction[ cols_to_unstack ]] , axis =1) ,id_vars =[ 'client_id ', 'card_id '] , value_vars = cols_to_unstack , value_name ='itemname ') [[ 'client_id ', 'card_id ', 'itemname ']]

print(df_long)
#################
#Question 2.1
#################

# Ils ont besoin d'un format transactionnel contenant un esemble d'items

#transaction['day'] = transaction['date'].dt.date
#transaction_mcc = transaction.groupby(['client_id','day'])['mcc'].agg(lambda x  : x.unique().tolist()).tolist()

#################
#Question 2.2
#################

