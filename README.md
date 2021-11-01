# Image Segmentation using U-Net
(1) patchify image and mask : DeepRoadStitching으로 만든 노면 파노라마 이미지를 patch로 만드는 코드(이미지 데이터의 사이즈가 너무 크므로 960X960 사이즈의 patch로 분할한 후, 512X512 사이즈로 최종 resizing)

(2) Pick mask : 포장 보수부 없는 patch 데이터 삭제 -> 최종 데이터 4224개 (보수부가 패치된 이미지의 3분의2 이상 차지하는 데이터 17개 제거) 
