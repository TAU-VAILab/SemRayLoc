import torch
from modules.semantic.semantic_mapper import room_type_to_id, zind_room_type_to_id

def predict_room_and_get_polygons(room_logits, room_polygons_all, room_selection_threshold, is_zind=False):
    """
    Predicts the room from logits and returns the corresponding polygons if confidence is high.
    """
    class_probs = torch.softmax(room_logits, dim=1)
    max_prob, room_idx = torch.max(class_probs, dim=1)

    room_polygons_filtered = []
    if max_prob.item() > room_selection_threshold:
        if is_zind:
            id_to_room_type = {v: k for k, v in zind_room_type_to_id.items()}
        else:
            id_to_room_type = {v: k for k, v in room_type_to_id.items()}
        predicted_room = id_to_room_type.get(room_idx.item())
        if predicted_room:
            # retrieve polygons for the predicted room
            room_polygons_filtered = room_polygons_all.get(predicted_room, [])
    
    return room_polygons_filtered 