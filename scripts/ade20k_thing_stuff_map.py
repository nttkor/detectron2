# ADE20K 150 Classes - Thing vs Stuff Manual Classification
# 사용자가 수정 가능한 분류 딕셔너리
# True = Thing (객체), False = Stuff (배경)

ADE20K_THING_STUFF_CLASSES = {
    # ==================== STUFF (배경 요소) ====================
    0: False,   # wall - 벽
    1: False,   # building - 건물
    2: False,   # sky - 하늘
    3: False,   # floor - 바닥
    4: False,   # tree - 나무 (개별 나무는 Thing이지만, 일반적으로 숲은 Stuff)
    5: False,   # ceiling - 천장
    6: False,   # road - 도로
    7: False,   # grass - 잔디
    8: False,   # sidewalk - 인도
    9: False,   # earth - 땅
    10: False,  # mountain - 산
    11: False,  # water - 물
    13: False,  # sea - 바다
    14: False,  # carpet - 카펫 (깔린 상태)
    17: False,  # plant - 식물 (일반)
    21: False,  # wall (brick) - 벽돌 벽
    24: False,  # snow - 눈
    25: False,  # path - 길
    26: False,  # runway - 활주로
    27: False,  # sand - 모래
    28: False,  # field - 들판
    29: False,  # ground - 지면
    31: False,  # dirt - 흙
    32: False,  # rock - 바위
    36: False,  # river - 강
    49: False,  # lake - 호수
    61: False,  # stairway - 계단 (구조물)
    72: False,  # clouds - 구름
    82: False,  # bridge - 다리 (구조물)
    91: False,  # hill - 언덕
    
    # ==================== THING (객체 요소) ====================
    12: True,   # person - 사람
    15: True,   # chair - 의자
    16: True,   # car - 차
    18: True,   # door - 문
    19: True,   # table - 테이블
    20: True,   # car (duplicate) - 차
    22: True,   # armchair - 안락의자
    23: True,   # seat - 좌석
    30: True,   # windowpane - 창문
    33: True,   # cabinet - 캐비닛
    34: True,   # cushion - 쿠션
    35: True,   # sofa - 소파
    37: True,   # poster - 포스터
    38: True,   # stage - 무대
    39: True,   # pool table - 당구대
    40: True,   # bed - 침대
    41: True,   # lamp - 램프
    42: True,   # truck - 트럭
    43: True,   # mirror - 거울
    44: True,   # clock - 시계
    45: True,   # bookcase - 책장
    46: True,   # chest of drawers - 서랍장
    47: True,   # wardrobe - 옷장
    48: True,   # sink - 싱크대
    50: True,   # toilet - 변기
    51: True,   # refrigerator - 냉장고
    52: True,   # microwave - 전자레인지
    53: True,   # oven - 오븐
    54: True,   # dishwasher - 식기세척기
    55: True,   # washer - 세탁기
    56: True,   # fan - 선풍기
    57: True,   # screen door - 방충망 문
    58: True,   # shower - 샤워기
    59: True,   # radiator - 라디에이터
    60: True,   # bottle - 병
    62: True,   # desk - 책상
    63: True,   # ottoman - 발받침대
    64: True,   # box - 상자
    65: True,   # column - 기둥
    66: True,   # signboard - 간판
    67: True,   # shelf - 선반
    68: True,   # fireplace - 벽난로
    69: True,   # van - 밴
    70: True,   # bus - 버스
    71: True,   # television receiver - TV
    73: True,   # book - 책
    74: True,   # computer - 컴퓨터
    75: True,   # swivel chair - 회전의자
    76: True,   # boat - 보트
    77: True,   # arcade machine - 아케이드 기계
    78: True,   # bench - 벤치
    79: True,   # countertop - 조리대
    80: True,   # stove - 스토브
    81: True,   # palm - 야자수
    83: True,   # base - 받침대
    84: True,   # trade name - 상표명
    85: True,   # buffet - 뷔페대
    86: True,   # flag - 깃발
    87: True,   # pillow - 베개
    88: True,   # screen - 스크린
    89: True,   # blanket - 담요
    90: True,   # apparel - 의류
    92: True,   # sconce - 벽등
    93: True,   # vase - 꽃병
    94: True,   # traffic light - 신호등
    95: True,   # tray - 쟁반
    96: True,   # ashcan - 쓰레기통
    97: True,   # plaything - 장난감
    98: True,   # swimming pool - 수영장
    99: True,   # barrel - 통
    100: True,  # basket - 바구니
    101: True,  # waterfall - 폭포
    102: True,  # tent - 텐트
    103: True,  # bag - 가방
    104: True,  # minibike - 미니바이크
    105: True,  # cradle - 요람
    106: True,  # oven (duplicate) - 오븐
    107: True,  # ball - 공
    108: True,  # food - 음식
    109: True,  # step - 계단 (개별)
    110: True,  # tank - 탱크
    111: True,  # trade name (duplicate) - 상표명
    112: True,  # microwave (duplicate) - 전자레인지
    113: True,  # pot - 냄비
    114: True,  # animal - 동물
    115: True,  # bicycle - 자전거
    116: True,  # dishwasher (duplicate) - 식기세척기
    117: True,  # screen (duplicate) - 스크린
    118: True,  # sculpture - 조각상
    119: True,  # hood - 후드
    120: True,  # sconce (duplicate) - 벽등
    121: True,  # jar - 항아리
    122: True,  # pipe - 파이프
    123: True,  # pitcher - 투구
    124: True,  # case - 케이스
    125: True,  # flower - 꽃
    126: True,  # canopy - 캐노피
    127: True,  # ashtray - 재떨이
    128: True,  # painting - 그림
    129: True,  # pedestal - 받침대
    130: True,  # fountain - 분수
    131: True,  # awning - 차양
    132: True,  # apparel (duplicate) - 의류
    133: True,  # pole - 기둥/폴
    134: True,  # bannister - 난간
    135: True,  # escalator - 에스컬레이터
    136: True,  # booth - 부스
    137: True,  # television receiver (duplicate) - TV
    138: True,  # airplane - 비행기
    139: True,  # conveyer belt - 컨베이어 벨트
    140: True,  # crt screen - CRT 스크린
    141: True,  # light - 조명
    142: True,  # poster (duplicate) - 포스터
    143: True,  # towel - 수건
    144: True,  # signboard (duplicate) - 간판
    145: True,  # plaything (duplicate) - 장난감
    146: True,  # traffic light (duplicate) - 신호등
    147: True,  # fan (duplicate) - 선풍기
    148: True,  # boat (duplicate) - 보트
    149: True,  # bar - 바
}

# 기본값 설정 함수 (딕셔너리에 없는 클래스 처리)
def get_thing_status(label_id, default=False):
    """
    label_id가 Thing인지 Stuff인지 반환
    
    Args:
        label_id (int): ADE20K 클래스 ID (0-149)
        default (bool): 딕셔너리에 없을 경우 기본값
    
    Returns:
        bool: True = Thing, False = Stuff
    """
    return ADE20K_THING_STUFF_CLASSES.get(label_id, default)
