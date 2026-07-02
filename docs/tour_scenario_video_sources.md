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

## 新增推荐候选：导览场景 + 地图参考

优先选择 Internet Archive 素材，因为它们可以直接下载，适合稳定复现实验；YouTube 素材可以作为界面参考或手动补充。

| 优先级 | 场景 | 视频 | 时长/大小 | 地图参考 | 为什么适合展示导览 |
| --- | --- | --- | --- | --- | --- |
| 1 | 景区/公园一镜到底 | [Flores Park 360 Walking Tour](https://archive.org/details/coriaca-Flores_Park_360_Walking_Tour_1020_W_Etiwanda_Avenue_Rialto_CA) / [MP4 下载](https://archive.org/download/coriaca-Flores_Park_360_Walking_Tour_1020_W_Etiwanda_Avenue_Rialto_CA/Flores_Park_360_Walking_Tour_1020_W_Etiwanda_Avenue_Rialto_CA.mp4) | 4:10，约 24 MB，854x480；另有 1080p 原文件约 128 MB | [OSM: Flores Park](https://www.openstreetmap.org/search?query=Flores%20Park%201020%20W%20Etiwanda%20Avenue%20Rialto%20CA) | 时长适中、路线清楚、包含 playground/picnic/walking trail 等多个导览点，最适合作为主 demo。 |
| 2 | 快速课堂演示 | [Woodbury 360 - Central Park](https://archive.org/details/cowomn-Woodbury_360_-_Central_Park) / [MP4 下载](https://archive.org/download/cowomn-Woodbury_360_-_Central_Park/Woodbury_360_-_Central_Park.mp4) | 1:30，约 8.8 MB，853x480 | [OSM: Central Park Woodbury](https://www.openstreetmap.org/search?query=Central%20Park%20Woodbury%20MN) | 文件很小，适合现场快速上传、跑完整 pipeline、演示自动导览模式。 |
| 3 | 室内/展馆导览 | [Seattle in 360: Asian Pacific American museum](https://archive.org/details/sc21wa-Seattle_in_360_-_Exploring_International_District_s_Asian_Pacific_American_museum) / [MP4 下载](https://archive.org/download/sc21wa-Seattle_in_360_-_Exploring_International_District_s_Asian_Pacific_American_museum/Seattle_in_360_-_Exploring_International_District_s_Asian_Pacific_American_museum.mp4) | 9:50，约 50 MB，854x480 | [OSM: Wing Luke Museum](https://www.openstreetmap.org/search?query=Wing%20Luke%20Museum%20Seattle) | 更像真实博物馆参观，适合展示“导览点覆盖率”和用户视角是否跟随推荐视角；地图可先定位到场馆，室内路线可用导览点表格补足。 |
| 4 | 古迹/景区导览 | [Ranch House at Los Penasquitos Canyon Preserve](https://archive.org/details/cg_0960-360-Degree_Tour_-_The_Ranch_House_At_Los_Penasquitos_Canyon_Preserve) / [MP4 下载](https://archive.org/download/cg_0960-360-Degree_Tour_-_The_Ranch_House_At_Los_Penasquitos_Canyon_Preserve/360-Degree_Tour_-_The_Ranch_House_At_Los_Penasquitos_Canyon_Preserve.mp4) | 7:55，约 47 MB，640x360；另有 720p 原文件约 26 MB | [OSM: Los Penasquitos Ranch House](https://www.openstreetmap.org/search?query=Los%20Penasquitos%20Canyon%20Preserve%20Ranch%20House) | 户外古迹导览感强，适合展示路线平滑度、舒适度、关键片段摘要和 2D 最终输出。 |

## YouTube 备选素材

这些素材视觉质量更好，但下载可能受 YouTube 限制；可以作为手动上传素材或界面参考。

| 场景 | 视频 | 地图参考 | 适合点 |
| --- | --- | --- | --- |
| 古迹/景区 | [Changdeokgung 360 VR Seoul](https://www.youtube.com/watch?v=T5DEWta0CCk) | [OSM: Changdeokgung Palace](https://www.openstreetmap.org/search?query=Changdeokgung%20Palace%20Seoul) | 宫殿景区、路线和地标明显，适合做“景点导览点”示例。 |
| 博物馆 | [Chicago in 360 - Field Museum](https://www.youtube.com/watch?v=j0HUJ3_QN_o) | [OSM: Field Museum Chicago](https://www.openstreetmap.org/search?query=Field%20Museum%20Chicago) | 室内展馆，适合展示推荐视角误差和用户观看轨迹。 |
| 园林/景区 | [Huntington Library and Gardens 360](https://www.youtube.com/watch?v=pecCTv7F-C4) | [OSM: Huntington Library and Gardens](https://www.openstreetmap.org/search?query=Huntington%20Library%20and%20Gardens%20San%20Marino) | 园林空间丰富，适合做路线舒适度和导览点覆盖展示。 |
| 图书馆/建筑参观 | [Seattle Central Library 360 Walking Tour](https://www.youtube.com/watch?v=mZlmkQJsrsU) | [OSM: Seattle Central Library](https://www.openstreetmap.org/search?query=Seattle%20Central%20Library) | 长视频，适合压力测试完整视频处理和长路线摘要。 |
| 大学博物馆 | [Eskenazi Museum of Art 360 Tour](https://www.youtube.com/watch?v=mdCm-5wMHW8) | [OSM: Eskenazi Museum of Art](https://www.openstreetmap.org/search?query=Eskenazi%20Museum%20of%20Art%20Indiana%20University) | 校园 + 博物馆结合，和课程大作业场景贴合。 |

## 真正 3D/VR180 备选

当前系统最适合处理 360° equirectangular 视频；真正的 3D/VR180 通常不是完整 360°，更适合作为后续“立体播放模式”扩展，而不是当前主 pipeline 的主素材。

| 场景 | 视频 | 地图参考 | 备注 |
| --- | --- | --- | --- |
| 博物馆 | [Louvre in 8K VR180 3D](https://www.youtube.com/watch?v=jOFQtSi14ik) | [OSM: Louvre Museum](https://www.openstreetmap.org/search?query=Louvre%20Museum%20Paris) | 视觉震撼，但偏 VR180，不适合直接做完整 360°推荐视角。 |
| 博物馆 | [Getty Center Museum Tour VR180 3D](https://www.youtube.com/watch?v=dJJgjvVlAb8) | [OSM: Getty Center](https://www.openstreetmap.org/search?query=Getty%20Center%20Los%20Angeles) | 可作为后续“3D/VR180 播放兼容”创新点。 |
| 博物馆 | [British Museum 3D VR180 walking clip](https://www.youtube.com/watch?v=rVgrc_zLy4Y) | [OSM: British Museum](https://www.openstreetmap.org/search?query=British%20Museum%20London) | 短素材，适合测试立体视频 UI，但不是当前主 demo 首选。 |

## 使用建议

1. 答辩现场优先用 `tamu_memorial_student_center_360_tour.mp4`、`rockwood_library_maker_space_360_tour.mp4` 或 Woodbury Central Park，处理速度快，场景清楚。
2. 展示“长视频摘要”和“路线舒适度”时用 `joe_sampson_park_360_walking_tour.mp4`、Flores Park 或 Los Penasquitos Ranch House，它们更像真实的一镜到底导览路线。
3. 展示“展馆/室内导览”时用 Wing Luke Museum 或 Field Museum，地图可以先定位到建筑，再用导览点表格/缩略图表达室内路线。
4. 地图功能可以采用“OSM 地图链接 + 导览点表格 + ERP 全景路线图”的轻量方案；后续再把导览点编号、时间段、推荐 yaw/pitch 和转向角画到地图或 ERP 全景缩略图上。
5. 当前播放器更适合普通 360°视频。严格的双目 3D VR 视频通常是 SBS/TB 或 VR180 格式，后续如果需要展示真正 3D 立体效果，需要再增加立体显示模式。
