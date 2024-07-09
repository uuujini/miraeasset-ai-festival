import dart_fss as dart

# Open DART API KEY 설정
api_key='698e1025376cfbf1631574822d5744a07b1c30ea'
dart.set_api_key(api_key=api_key)

#검색정보 쿼리
company = '삼성전자'

# DART 에 공시된 회사 리스트 불러오기
corp_list = dart.get_corp_list()

# 삼성전자 검색
corpName = corp_list.find_by_corp_name(company, exactly=True)[0]

print('연결재무제표 검색완료')

# 2012년부터 연간 연결재무제표 불러오기
fs = corpName.extract_fs(bgn_de='20200101')

print('연결재무제표 데이터 호출 완료')

# 재무제표 검색 결과를 엑셀파일로 저장 ( 기본저장위치: 실행폴더/fsdata )
fs.save()
print('연결재무제표 저장완료')
