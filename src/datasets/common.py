def merge_with_common_ids_comprehensive(dict1, dict2, user_map1, user_map2):
    '''
    merge two datasets into one
    id1: list[items1], id2: list[items2] -> real_id: (id1, list[items1], id2, list[items2])
    '''
    # Find common IDs
    common_ids = set(user_map1.keys()) & set(user_map2.keys())
    num_common_ids = len(common_ids)
    
    # Create the merged result
    result = {}

    for common_id in common_ids:
        id1, id2 = user_map1[common_id], user_map2[common_id]
        result[common_id] = (id1, dict1[id1], id2, dict2[id2])
    
    return result, num_common_ids