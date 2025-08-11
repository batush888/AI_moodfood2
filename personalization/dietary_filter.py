class DietaryFilter:
    def __init__(self):
        self.restriction_rules = self._load_restriction_rules()
        self.substitution_database = self._load_substitutions()
        
    def filter_recommendations(self, recommendations, dietary_restrictions):
        filtered_recs = []
        
        for rec in recommendations:
            if self.is_compatible(rec, dietary_restrictions):
                filtered_recs.append(rec)
            else:
                # Try to create compatible version
                adapted_rec = self.create_compatible_version(rec, dietary_restrictions)
                if adapted_rec:
                    filtered_recs.append(adapted_rec)
        
        return filtered_recs
    
    def create_compatible_version(self, food_item, restrictions):
        adapted_item = food_item.copy()
        
        for restriction in restrictions:
            if restriction in self.substitution_database:
                substitutions = self.substitution_database[restriction]
                
                # Apply ingredient substitutions
                for ingredient in adapted_item.ingredients:
                    if ingredient.conflicts_with(restriction):
                        substitute = substitutions.get(ingredient.name)
                        if substitute:
                            adapted_item.replace_ingredient(ingredient, substitute)
                        else:
                            return None  # Cannot adapt
        
        return adapted_item