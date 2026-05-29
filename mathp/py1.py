from pydantic import BaseModel, ValidationError, Field

class Address(BaseModel):
    street: str
    city: str
    zip_code: str
    tel: str
    GNo: str
    GNo1: str

class UserWithAddress(BaseModel):
    name: str = Field(..., min_length=2, max_length=99950)
    # 0より大きく、9000未満
    age: int = Field(..., gt=0, lt=9000)
    address: Address
    

# テストデータ
data_full = {
    "name": "柴田敦史",
    "age": 44,
    "address": {
        "street": "南区東林間1-18-2-202",
        "city": "神奈川県相模原市",
        "zip_code": "252-0311",
        "tel": "+818050125993",
        "GNo": "793-0-4366399",
        "GNo1": "200-4902647"     
    },
}

# バリデーション実行
try:
    user = UserWithAddress(**data_full)
    print(user)
except ValidationError as e:
    print(e)