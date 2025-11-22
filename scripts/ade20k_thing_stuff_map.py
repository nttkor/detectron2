# ADE20K 공식 Thing/Stuff 분류
# 출처: CSAILVision MIT - ADE20KInfo150.csv
# Stuff=1 (배경), Thing=0 (객체)

import csv
import os

def load_ade20k_official_mapping():
    """
    ADE20KInfo150.csv를 읽어서 Thing/Stuff 매핑 생성
    
    Returns:
        dict: {class_id: is_thing} (0-based index)
        dict: {class_id: class_name}
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'ADE20KInfo150.csv')
    
    is_thing_map = {}
    id2label = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 건너뛰기
        
        for row in reader:
            idx = int(row[0])  # 1-based (1-150)
            stuff = int(row[4])  # 1=Stuff, 0=Thing
            name = row[5]
            
            # 0-based로 변환 (0-149)
            class_id = idx - 1
            
            # Thing/Stuff 변환: CSV에서 Stuff=1, Thing=0
            # 우리 형식: Thing=True, Stuff=False
            is_thing = (stuff == 0)
            
            is_thing_map[class_id] = is_thing
            id2label[class_id] = name
    
    return is_thing_map, id2label

# 전역 변수로 로드
ADE20K_THING_STUFF_CLASSES, ADE20K_CLASS_NAMES = load_ade20k_official_mapping()

if __name__ == "__main__":
    # 테스트
    thing_count = sum(1 for v in ADE20K_THING_STUFF_CLASSES.values() if v)
    stuff_count = sum(1 for v in ADE20K_THING_STUFF_CLASSES.values() if not v)
    
    print(f"ADE20K 공식 분류 (CSAILVision)")
    print(f"Thing: {thing_count}개")
    print(f"Stuff: {stuff_count}개")
    print(f"\n처음 10개 클래스:")
    for i in range(10):
        thing_or_stuff = "Thing" if ADE20K_THING_STUFF_CLASSES[i] else "Stuff"
        print(f"  {i}: {ADE20K_CLASS_NAMES[i]:<30} [{thing_or_stuff}]")
