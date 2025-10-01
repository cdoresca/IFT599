import pandas as pd
from typing import List, Set

class Recommendation:
    def __init__(self, rules: pd.DataFrame):
        """
        Initialise le système de recommandation avec les règles d'association.
        
        rules : DataFrame avec les colonnes 'antecedents', 'consequents', 'confidence', 'lift'
        """
        self.rules = rules[(rules['lift'] >= 1.01) & (rules['confidence'] >= 0.3) & 
                           (rules['support'] >= 0.01) &(rules['leverage'] > 0.001)]

    def recommend(self, products: Set[str], n: int = 5) -> List[str]:
        """
        Recommande des produits en fonction des produits déjà présents.
        
        products : ensemble des produits déjà achetés
        n : nombre maximal de produits à recommander
        
        Retourne une liste de produits recommandés.
        """
        # filtrer les règles dont les antécédents sont un sous-ensemble des produits donnés
        mask = self.rules['antecedents'].apply(lambda x: x.issubset(products))
        relevant_rules = self.rules[mask]

        # si aucune règle pertinente, retourner une liste vide
        if relevant_rules.empty:
            return ['empty']

        # trier les règles par confiance puis par lift de manière décroissante
        relevant_rules = relevant_rules.sort_values(['confidence', 'lift'], ascending=False)

        # extraire les produits consécutifs (consequents) et supprimer ceux déjà présents
        recommended_products = (
            product
            for consequent in relevant_rules['consequents']
            for product in consequent
            if product not in products
        )

        # construire la liste finale des recommandations, en s'assurant qu'ils sont uniques
        seen = set()
        recommendations = []
        for product in recommended_products:
            if product not in seen:
                seen.add(product)
                recommendations.append(product)
                # arrêter dès que l'on a atteint le nombre n souhaité
                if len(recommendations) >= n:
                    break

        return recommendations