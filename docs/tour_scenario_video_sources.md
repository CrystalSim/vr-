# Tour Scenario 360 Video Sources

本清单面向“沉浸式校园/景区/展厅导览”的后续优化，用于测试导览点提取、推荐视角、观看轨迹、舒适度评价和导览地图展示。

## 已下载素材

| 场景 | 本地文件 | 时长 | 分辨率 | 适合演示 | 来源 | 地图参考 |
| --- | --- | ---: | --- | --- | --- | --- |
| 校园/学生中心导览 | `data/raw/videos/tour_scenarios/tamu_memorial_student_center_360_tour.mp4` | 1:38 | 854x480 | 校园开放日、学生中心、室内导览点 | [Internet Archive](https://archive.org/details/tamutx-Memorial_Student_Center_360_Tour-20200115) | [OpenStreetMap 搜索](https://www.openstreetmap.org/search?query=Memorial%20Student%20Center%20Texas%20A%26M) |
| 图书馆/创客空间导览 | `data/raw/videos/tour_scenarios/rockwood_library_maker_space_360_tour.mp4` | 2:44 | 854x480 | 展厅/实验室/创客空间参观 | [Internet Archive](https://archive.org/details/mecm-Rockwood_Library_Maker_Space_360_Tour) | [OpenStreetMap 搜索](https://www.openstreetmap.org/search?query=Rockwood%20Library%20Maker%20Space%20Gresham%20OR) |
| 景区/公园步行导览 | `data/raw/videos/tour_scenarios/joe_sampson_park_360_walking_tour.mp4` | 6:39 | 854x480 | 一镜到底路线、景点覆盖、路线舒适度 | [Internet Archive](https://archive.org/details/coriaca-Joe_Sampson_Park_360_Walking_Tour_650_W._Randall_Avenue_Rialto_CA) | [OpenStreetMap 搜索](https://www.openstreetmap.org/search?query=Joe%20Sampson%20Park%20650%20W.%20Randall%20Avenue%20Rialto%20CA) |

## 可选候选

这些视频也适合场景，但文件较大或平台下载受限，可以作为后续手动补充：

| 场景 | 视频 | 说明 |
| --- | --- | --- |
| 古迹/景区导览 | [Pyramid Of Unas And The Pyramid Texts 360 VR Tour](https://archive.org/details/pyramid-of-unas-and-the-pyramid-texts-360-vr-tour-sam-mayfair-1080p-h-264) | 约 5:56，1080p，约 178 MB，适合做高质量景区/博物馆导览示例。 |
| 博物馆导览 | [The Louvre Museum Guided Tour in 360 VR](https://www.youtube.com/watch?v=xdiTy_YjVvg) | 约 1:48，适合展示展馆路线，但命令行下载时被 YouTube 登录验证拦截。 |
| 校园导览 | [MIT 360 VR Walking Tour](https://www.youtube.com/watch?v=aFTPoaNsWtM) | 约 40:52，适合长视频摘要压力测试，但较大且 YouTube 下载受限。 |

## 使用建议

1. 答辩现场优先用 `tamu_memorial_student_center_360_tour.mp4` 或 `rockwood_library_maker_space_360_tour.mp4`，处理速度快，场景清楚。
2. 展示“长视频摘要”和“路线舒适度”时用 `joe_sampson_park_360_walking_tour.mp4`，它更像真实的一镜到底导览路线。
3. 地图功能可以先采用“地图链接 + 导览点表格”的轻量方案；后续再把导览点编号、时间段、推荐 yaw/pitch 和转向角画到地图或 ERP 全景缩略图上。
4. 当前播放器更适合普通 360°视频。严格的双目 3D VR 视频通常是 SBS/TB 格式，后续如果需要展示真正 3D 立体效果，需要再增加立体显示模式。
