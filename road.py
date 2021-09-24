import cv2 as cv
import numpy as np
import time
vid = cv.VideoCapture("roadgazebo.avi")

if vid.isOpened() == False:
    print("Cannot open input video")
    exit()

imgRoadMin = [210, 320]  #размеры сжатого в последующем изображении, с которым будем работать (высота 240 px, ширина 320px, то есть вдвое меньше оригинала

roi = np.float32([[10, 200], #массив точек области интереса дорожной разметки x, y начиная отсчет с левого верхнего угла (float32 т.е. элементы массива с плавающей точкой с точностью до 8 знака после запятой 1.0000001) и по часовой стрелки по - порядку
                  [300, 200], #тип float32 для трансформации скорее всего???
                  [230, 140],
                  [85, 140]])
                  # [20, 200],  левая нижняя точка
                  # [310, 200], правая нижняя точка
                  # [225, 120], правая верхняя точка
                  # [100, 120]] левая верхняя точка

roi_draw = np.array(roi, dtype=np.int32) #копируем массив с областью интереса для рисования с преобразованием данных к целочисленному int32. Целые числа в диапазоне от -2147483648 по 2147483647, (числа размером 4 байта).
                                         #первый аргумент это массив, который копируем, второй - тип данных

dst = np.float32([[0, imgRoadMin[0]],            #массив координат области начиная с левого нижнего угла по часовой стрелке откладывая от вернхнего левого, куда поместим область интереса, это по сути углы исходного ужатого изображения
                  [imgRoadMin[1], imgRoadMin[0]],
                  [imgRoadMin[1], 0],
                  [0, 0]])
key = 0
while (cv.waitKey(40) != 27): #если нужно переключение кадров сделать по клавише, в скобках cv.waitKey() пусто, а 40 это 40 мс между кадрами то есть фпс 25
    # // waitKey
    # возвращает - 1, если ничего не нажать
    # // если нажали клавишу, эта функция возвращает код ключа ASCII
    # // if (waitKey() == 27) в таком случае будем ждать нажатия клавыши esc (код 27) для выхода
    # // и если оставим пустым аргумент waitKey, то кадр будет обновляться с нажатием любой клавиши
    # // Число 40 получается по нехитрой формуле:
    # // 1000 миллисекунд / 25 кадров в секунду = 40 миллисекунд
    # // чаще обращаться смысла нет, если камера не поддерживает большее fps
    ret, frame = vid.read()
    if ret == False:
        print("End of video")
        #cap.release()
        #cap = cv.VideoCapture(r"test_videos/output1280.avi")
        #ret, frame = cap.read()
        #break

    resized = cv.resize(frame, (imgRoadMin[1], imgRoadMin[0])) #уменьшаем исходный кадр до заданых размеров для более быстрой работы обработки изображения (в частности бинаризации)
    cv.imshow("frame", resized)

    r_channel = resized[:, :, 2]
    binary = np.zeros_like(r_channel)
    binary[(r_channel > 200)] = 1
    #cv.imshow("r_channel",binary)

    #альтернативная бинаризация по красной компоненте
    # hls=cv.cvtColor(resized, cv.COLOR_BGR2HLS)
    # s_channel = resized[:, :, 2]
    # binary2 = np.zeros_like(s_channel)
    # binary2[(r_channel > 160)] = 1
    #
    # allBinary = np.zeros_like(binary)
    # allBinary[((binary == 1)|(binary2 == 1))] = 255

    #более короткая бинаризация изображения через treshold с предварительным переводом изображения в одноканальное
    resized = cv.cvtColor(resized, cv.COLOR_RGB2GRAY)
    ret, allBinary = cv.threshold(resized, 150, 255, cv.THRESH_BINARY) #для игнорирования флага наличия изображения (_, allBinary)
    #print(allBinary.shape[-1]) #вывод каналов изображения
    #cv.imshow("binary", allBinary)

    allBinary_visual = allBinary.copy() #предварительно копируем из основного изображения в новое, чтобы не испортить исходное
    cv.polylines(allBinary_visual, [roi_draw], True, 255) #соеднияем точки массива [roi_draw] одной линией белой линией для одноканального изображения 255
    cv.imshow("roi", allBinary_visual)

    #расчет матрицы преобразований, где первый аргумент это массив с координатами области интрерса, то есть наша трапеция
    #а второй аргумент как раз координаты углов ужатого изображения
    M = cv.getPerspectiveTransform(roi, dst) #на выходе матрица M для получения прямоугольного изображения

    #https://youtu.be/ApUQ0EgrnM0
    #https://learnopencv.com/feature-based-image-alignment-using-opencv-c-python/
    #https://www.geeksforgeeks.org/perspective-transformation-python-opencv/
    # allBinary - изображение, которое хотим преобразовать, M - матрица преобразования, далее размер изображения (как и входное),
    #flags=cv.INTER_LINEAR - способ расчета
    # imgIntpl преобразование изображение из трапецевидного в обычное квадратное
    imgIntpl = cv.warpPerspective(allBinary, M, (imgRoadMin[1], imgRoadMin[0]), flags=cv.INTER_LINEAR)
    #cv.imshow("imgIntpl", imgIntpl)

    #поиск самых белых столбцов на преобразованном из трапеции в прямоугольник изображении
    #укороченный вариант для суммирования, где axis=0 это вертикаль и в histogram мы получим сумму белых элементов в каждом столбце
    #суммирование пикселей происходит от низа картинки до середины imgIntpl.shape[0]//2
    histogram = np.sum(imgIntpl[imgIntpl.shape[0] // 2:, :], axis=0)
    #поиск начинается с середины по вертикали и смотрим слева и справа
    midpoint = histogram.shape[0]//2 #номер центрального столбца, берем целое число обазательно, поэтому //, а не /
    idLeft = np.argmax(histogram[:midpoint])
    idRight = np.argmax(histogram[midpoint:]) + midpoint

    #альтернатиным более долгим вариантом: просуммируем все белые пиксели в каждом стобце через for
    #shape[0] - строка картинки shape[1] столбец картинки
    #находим номер пикселя по X самого белого столбца в левой части области интереса и в правой области интереса
    # indexMas = 0
    # whitePix = 0
    # idLeft = 0
    # idRight = 0
    # maxWhitePix = 0
    # arrayWhitePixel = []
    # for i in range(0, imgIntpl.shape[1]):  # по правой части кадра
    #     indexMas = indexMas + 1
    #     whitePix = 0
    #     if (i == imgIntpl.shape[1] // 2):
    #         maxWhitePix = 0
    #     for j in range(imgIntpl.shape[0] // 2, imgIntpl.shape[0]):
    #         if (imgIntpl[j, i] > 100):
    #             whitePix = whitePix + 1
    #         if (whitePix > maxWhitePix):
    #             if (i < imgIntpl.shape[1] // 2):  # левая часть
    #                 idLeft = i
    #             else:
    #                 idRight = i
    #             maxWhitePix = whitePix

    imgIntpl_visual = imgIntpl.copy() #рисовть серые линии будем на копии изображении для демонстрации чтобы не испортить оригинал
    cv.line(imgIntpl_visual, (idLeft, imgIntpl_visual.shape[0]//2), (idLeft, imgIntpl_visual.shape[0]), 122, 3)
    cv.line(imgIntpl_visual, (idRight, imgIntpl_visual.shape[0]//2), (idRight, imgIntpl_visual.shape[0]), 122, 3)
    cv.imshow("imgIntpl_visual", imgIntpl_visual)

    wind = 7 #количество окон для поиска белой линии
    windH = np.int(imgIntpl.shape[0]/wind) #высота окна с учетом их количества и высоты исходного изображения (чтобы сверху до низу были окна)
    windSearchWidth = 25 #ширина окна в пикселях

    xCentrLeftWind = idLeft #центр первого белого окна как раз совпадает с номером столбца
    xCenRightWind = idRight

    leftLinePixIndex = np.array([], dtype = np.int16) #создаем пока пустой массив центров линии с типом данных int16, т.к. функции работают с этим типом
    rightLinePixIndex = np.array([], dtype = np.int16) #справа по аналогии

    # преобразование изображеня в трехканальное (оно всеравно будет ЧБ) но тогда можно будет рисовать цветные окна и линии на этом изображении
    #outImg = np.dstack((imgIntpl, imgIntpl, imgIntpl))

    # получаем номера (то есть индексы) всех белых пикселей на изоборажении
    nonZero = imgIntpl.nonzero()          #imgIntpl-изображение, то есть по факту массив, к которому применяем функцию, возвращающая список номеров всех ненулевых элементов массива imgIntpl
    #nonZero-в этому списке хранятся индексы по строкам и по столбцам всех ненулевых элементов массива
    whitePixY = np.array(nonZero[0]) #выделяем индексы строк белых пикселей уже в массив, а не в список
    whitePixX = np.array(nonZero[1]) #выделяем индексы столбцов белых пикселей уже в массив, а не в список

    sumX = 0
    for i in range(wind): #создаем окна для поиска центра линии в каждом окне
        #на каждом проходе for появляется по два окна снизу вверх (дебагом точки можно поставить на функции ректангл)

        #координаты углов окон (y координата для левго и правого она одинаковая, а x - меняется)
        windY1 = imgIntpl.shape[0] - (i + 1) * windH #верхняя координата от нижнего края изображения вычитаем (номер окна + 1)*на высоту окна
        windY2 = imgIntpl.shape[0] - (i) * windH     #нижняя координата также только без 1 как раз на одну высоту окна меньше

        leftWindX1 = xCentrLeftWind - windSearchWidth #меньшая координта: от центра окна отнимается половина ширины
        leftWindX2 = xCentrLeftWind + windSearchWidth #большая координта: к центру окна прибавляется половина ширины
        rightWindX1 = xCenRightWind - windSearchWidth
        rightWindX2 = xCenRightWind + windSearchWidth

        #отрисовываем окна поиска белой линии
        cv.rectangle(imgIntpl, (leftWindX1, windY1), (leftWindX2, windY2), 122, 1) # поставить outImg если хотим нарисовать цветные окна поиска cv.rectangle(outImg, (leftWindX1, windY1), (leftWindX2, windY2), (0, 255, 0), 1)
        cv.rectangle(imgIntpl, (rightWindX1, windY1), (rightWindX2, windY2), 122, 1)
        #cv.imshow("searchWind", imgIntpl) #вернуть outImg если преобразуем в трехканальное для цветной отрисовки окон поиска

        #ищем пиксели в каждом окне, принадлежащие разметке
        #если координата текущего белого пикселя попдает внутрь диапазона по X и Y
        #текущего скользящего окна (сначала левого, потом правого) то сохраним координаты пикселей в массив leftPixInWind
        #вконце обязательно указать [0], если этот нулевой индекс не укажем, будет вместо массива кортеж
        leftPixInWind = ((whitePixY >= windY1) & (whitePixY <= windY2) & (whitePixX >= leftWindX1) & (whitePixX <= leftWindX2)).nonzero()[0]
        rightPixInWind = ((whitePixY >= windY1) & (whitePixY <= windY2) & (whitePixX >= rightWindX1) & (whitePixX <= rightWindX2)).nonzero()[0]
        # *** поставить точку дебага и вывести массив номеров (то есть индексов) белых пикселей в окне
        #print(leftPixInWind)
        leftLinePixIndex = np.concatenate((leftLinePixIndex, leftPixInWind)) #на каждой новой этерации пополняем массив индексов - склейка(как инкрементирование)
        rightLinePixIndex = np.concatenate((rightLinePixIndex, rightPixInWind))

        #находим среднее значение белой линии чтобы расположить сооответсвенно окно, которое наглядно следит за линией
        #смещаем окно, если попало пикселей больше 40
        #и если в окно не попадает разметки или попадает меньше 40 то окно не смещается и
        #они получаются столбиками. Отчетливо это видно при прорисовки на пунктирной части разметки
        if len(leftPixInWind) > 40:
            #функция np.mean возвращает значение типа float, чтобы получить значения типа инт, преобразуем: np.int
            xCentrLeftWind = np.int(np.mean(whitePixX[leftPixInWind])) #индексы индексов пикселей. напомню, координаты пикселей, попавших в окно хранятся в whitePixX
            #***поставить точку для дебага и вывести показания расчитанного центра линии для очередного окна на основе элементов, попавших в окно
            #print(xCentrLeftWind)                                                           #индексы leftPixInWind, попавшие в мелькое следаящее окошко??
        if len(rightPixInWind) > 40:
            xCenRightWind = np.int(np.mean(whitePixX[rightPixInWind]))
            #print(xCenRightWind)

        #ПЕРВЫЙ СПОСОБ НАХОЖДЕНИЯ ЦЕНТРА ДОРОГИ (#рисуем центральну линию относительно найденных центров в каждом скользяцем окне)
        x = int((xCenRightWind+xCentrLeftWind)/2) #находим центр дороги по каждому данным середины линии каждого окна поиска
        #print(x)
        cv.circle(imgIntpl, (x, windY1), 3, 122, -1)  # 3-радиус, далее цвет и толщина линии (-1 залить круг)
        sumX = sumX + x
    # далее можно сложить все показания в массив и устреднить это значение. Тогда в текущем центре будет отражаться не только то что за копотом авто, а как и дальше трасса себя ведет
    cv.circle(imgIntpl, (int(sumX/wind), 100), 5, 122, -1)
    #перекрас пикселей, оказавшихся внутри скользящих окон (сначала нужно указать строки, затем столбцы)
    # outImg[whitePixY[leftLinePixIndex], whitePixX[leftLinePixIndex]] = [0, 255, 0]
    # outImg[whitePixY[rightLinePixIndex], whitePixX[rightLinePixIndex]] = [0, 0, 255]
    #cv.imshow("Lane", outImg)

    #ВТОРОЙ СПОСОБ ПОИСКА ЦЕНТР ДОРОГИ (рисуем параболу окружностью)
    # leftx = whitePixX[leftLinePixIndex]
    # lefty = whitePixY[leftLinePixIndex]
    # rightx = whitePixX[rightLinePixIndex]
    # righty = whitePixY[rightLinePixIndex]
    #
    # left_fit = np.polyfit(lefty, leftx, 2)
    # right_fit = np.polyfit(righty, rightx, 2)
    # # центр линии лежит между левой и правой линией разметки
    # center_fit = ((left_fit+right_fit)/2)
    #
    # for ver_ind in range(outImg.shape[0]):
    #     gor_ind = ((center_fit[0]) * (ver_ind ** 2) +
    #                 center_fit[1] * ver_ind +
    #                 center_fit[2])
    #     cv.circle(outImg, (int(gor_ind), int(ver_ind)), 2, (255, 0, 0), 1)

    cv.imshow("CenterLine", imgIntpl) # вернуть outImg, если пребразуем к трехканальному для рисования цветных квадратов
