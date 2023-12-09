## Dataset
### 1. Cora Dataset
- 논문 인용 네트워크 데이터
- Node: 논문을 나타냄. 총 2,708개
- Node features: 해당 논문에 나타나는 단어들의 존재 여부를 0과 1로 표시한 벡터
- Edge: 한 논문이 다른 논문을 인용함을 나타냄. 총 10,556개
- Class: 논문 주제를 나타냄. 총 7개

-------------------------------------------------------------------------------------------------

### 2. Amazon Dataset
- Photos, Computers로 나눠져 있으며 실습에선 Computers dataset 사용
- Node: 상품을 나타냄. 총 13,752개
- Node features: 상품 리뷰에 나타나는 단어들의 존재 여부를 0과 1로 표시한 벡터
- Edge: 두 상품이 자주 함께 구매됨을 나타냄. 총 491,722개
- Class: 제품 카테고리를 나타냄. 총 10개

-------------------------------------------------------------------------------------------------

### 3. Reddit Dataset
- 2014년 9월에 작성된 게시물 데이터
- Node: 게시물을 나타냄. 총 232,965개
- Node features: post title, post’s comment, score of post, number of comments on the post
- Edge: same user가 게시물에 comments를 남겼을 때 두 게시물 사이에 생성됨. 총 114,615,892개
- Class: 커뮤니티를 나타냄. 총 50개