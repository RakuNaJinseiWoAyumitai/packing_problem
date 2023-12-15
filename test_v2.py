import random
import matplotlib.pyplot as plt
import math
from matplotlib import patches
import numpy as np
import csv
import itertools
import copy

class paking_program:
    def __init__(self):
        # csvファイルを指定
        self.MyPath = 'E:\Reserch/programs/jakobs1.csv'
        # self.MyPath = 'E:\Reserch/programs/sample1.csv'
        # self.MyPath = 'E:\Reserch/programs/ex2.csv'
        self.piece_collision = [False, False, False, False]
        self.piece_MinMax = [[10000, 0, 10000, 0], [10000, 0, 10000, 0], [10000, 0, 10000, 0], [10000, 0, 10000, 0]]
        self.rows = []
        self.arr_piece = []
        self.poly_row = []
        self.False_list = []
        self.colors = ['blue', 'red', 'green', 'purple', 'Yellow', 'Black']

    # 辺の長さ
    def abs_liner(self, x1, x2):
        return np.sqrt(np.square(x2[0] - x1[0]) + np.square(x2[1] - x1[1]))

    def collision_rule(self, poly_list, r):  # 通信圏内検出 #円の大きさは要調整
        print("r =", r)
        collision = []
        for k in range(len(pa.poly_row)):
            for j in range(len(poly_list[k])):
                if (int(poly_list[k][j][0]) - 5) ** 2 + (int(poly_list[k][j][1]) - 5) ** 2 <= r ** 2:#円の中心は(5,5)
                    collision.append(poly_list[k])
                    break
        return collision

    def InputCheck(self, p1, p2):  # 結合チェック
        flag = True
        for i in range(len(p1)):  # ここのフラグを"予め接することがわかってる分"から引くことで，余分に接していることを確認するように変更
            for j in range(len(p2)):
                if i == len(p1) - 1 and j == len(p2) - 1:
                    if self.intersection(p1[i], p1[0], p2[j], p2[0]):
                        flag = False
                elif i == len(p1) - 1:
                    if self.intersection(p1[i], p1[0], p2[j], p2[j + 1]):
                        flag = False
                elif j == len(p2) - 1:
                    if self.intersection(p1[i], p1[i + 1], p2[j], p2[0]):
                        flag = False
                else:
                    if self.intersection(p1[i], p1[i + 1], p2[j], p2[j + 1]):
                        flag = False
        if flag:
            # print("flag=True1")
            # if (self.calc_area(self.Tolerance(p1.copy(), p2.copy()))) >= (
            #         self.calc_area(p1.copy()) + self.calc_area(p2.copy())):
            #     print("flag=TRUE")
            #     return True
            # else:
            #     return False
            return True
        else:
            # print(self.calc_area(self.Tolerance(p1.copy(),p2.copy())),self.calc_area(p1.copy()),self.calc_area(p2.copy()))
            if (self.calc_area(self.Tolerance(p1.copy(), p2.copy()))) >= (
                    self.calc_area(p1.copy()) + self.calc_area(p2.copy())):
                # print("flag=TRUE2")
                return True
            return False

    def plt_InputCheck(self, p1, p2):
        flag = False
        if(self.calc_area(self.Tolerance(p1.copy(), p2.copy()))) >= (
                    self.calc_area(p1.copy()) + self.calc_area(p2.copy())):
            # print("TRUE")
            flag = True

        for i in range(len(p1)):  # ここのフラグを"予め接することがわかってる分"から引くことで，余分に接していることを確認するように変更
            for j in range(len(p2)):
                if i == len(p1) - 1 and j == len(p2) - 1:
                    if self.intersection(p1[i], p1[0], p2[j], p2[0]):
                        flag = False
                elif i == len(p1) - 1:
                    if self.intersection(p1[i], p1[0], p2[j], p2[j + 1]):
                        flag = False
                elif j == len(p2) - 1:
                    if self.intersection(p1[i], p1[i + 1], p2[j], p2[0]):
                        flag = False
                else:
                    if self.intersection(p1[i], p1[i + 1], p2[j], p2[j + 1]):
                        flag = False

        if flag:
            # print("flag=True")
            return True
        else:
            return False

    # 交差判定
    def intersection(self, x1, x2, x3, x4):
        if (((x1[0] - x2[0]) * (x3[1] - x1[1]) + (x1[1] - x2[1]) * (x1[0] - x3[0])) * (
                (x1[0] - x2[0]) * (x4[1] - x1[1]) - (x1[1] - x2[1]) * (x1[0] - x4[0])) < 0):
            if (((x3[0] - x4[0]) * (x1[1] - x3[1]) + (x3[1] - x4[1]) * (x3[0] * x1[0])) * (
                    (x3[0] - x4[0]) * (x2[1] - x3[1]) + (x3[1] - x4[1]) * (x3[0] - x2[0])) < 0):
                return True  # 交差する
        return False  # 交差しない

#評価関数
    def search_func(self, p1, p2):
        p1_info = self.degree_liner_set(p1)  # 角度とそのなす辺について情報取得
        p2_info = self.degree_liner_set(p2)
        score = []
        max_score = 0
        candidateComb = []
        comb = []

        for i in range(len(p1_info)):
            for j in range(len(p2_info)):
                #角度(point1)
                total_degree = p1_info[i][0] + p2_info[j][0]
                #三角形の統合
                if len(p1_info) == 3 and len(p2_info) == 3:
                    # print("三角形の統合")
                    if 90 >= total_degree:
                        deg_rate = float(total_degree / 90)
                    else:
                        deg_rate = float(90 / total_degree)
                    if deg_rate <= 1:
                        point1 = deg_rate
                    else:
                        point1 = 0
                #多角形
                else:
                    degRate = []
                    deg = 360
                    while deg != 0:
                        if deg > 0:
                            if deg >= total_degree:
                                list_degRate = float(total_degree / deg)
                            else:
                                list_degRate = float(deg / total_degree)

                        if list_degRate <= 1:
                            degRate.append(list_degRate)
                        else:
                            degRate.append(0)
                        deg -= 90
                    point1 = max(degRate)

                #辺1
                if p1_info[i][1] <= p2_info[j][1]:
                    side_rate1 = float(p1_info[i][1] / p2_info[j][1])
                else:
                    side_rate1 = float(p2_info[j][1] / p1_info[i][1])
                #辺2
                if p1_info[i][2] <= p2_info[j][2]:
                    side_rate2 = float(p1_info[i][2] / p2_info[j][2])
                else:
                    side_rate2 = float(p2_info[j][2] / p1_info[i][2])
                #point2
                point2 = side_rate1 * side_rate2

                point = point1 * point2
                # print(point)
                count = 0
                if max_score < point:
                    candidateComb = []
                    max_score = point
                    candidateComb.append([int(i), int(j)])
                    # print("scoreProces", comb, max_score)
                elif max_score == point:
                    candidateComb.append([int(i), int(j)])

        if len(candidateComb) != 1:
            comb = random.choice(candidateComb)
        else:
            # print("[1]candidateComb:", candidateComb)
            comb = candidateComb[0]

        # print("comb, max_score :", comb, max_score)
        return comb, max_score

    #移動前と移動後のピースの面積に変化があるか調べる
    def piece_check(self, pp, mp):  #pp：移動前のピース，mp：移動後のピース
        pp_area = self.calc_area(pp)
        mp_area = self.calc_area(mp)
        #移動後のピースの面積が異なる場合False
        if mp_area != pp_area:
            return False
        else:
            return True

    def piece_move(self, piece, code):
        # print("code = ", code)
        p1 = piece[0]
        p2 = piece[1]
        p2_copy = copy.deepcopy(piece[1])
        a = p1[code[0]]
        b = p2[code[1]]
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        for i in range(len(p2)):
            p2[i][0] += dx
            p2[i][1] += dy

        flag = True
        reverseFlag = ["normal", "reverse"]
        degree = [0, 90, 180, 270]
        count = 0
        if self.InputCheck(p1.copy(), p2.copy()) != True:
            for i in range(len(reverseFlag)):
                rev_p2 = self.reversePiece(p2.copy(), reverseFlag[i], a)
                for j in range(len(degree)):
                    p = self.rotate_cie(rev_p2.copy(), degree[j], a)
                    for k in range(len(p)):
                        if p[k] in p1:
                            count += 1
                    # print("C,F,D,IP = ", count, reverseFlag[i], degree[j], self.InputCheck(p1.copy(), p.copy()))
                    # print("**p1**", p1)
                    # print("**p2**", p)
                    if count == len(p):
                        print("重なっている")
                        flag = False
                    count = 0
                    if self.InputCheck(p1.copy(), p.copy()) == True and flag:
                        if self.piece_check(p2_copy, p.copy()):
                            np = self.Tolerance(p1.copy(), p.copy())
                            if np != False:
                                if self.calc_check(p1.copy(), p.copy(), np.copy()):
                                    print("**^^**")
                                    return np
                        else:
                            print("**「変形」**")

            return False
        else:
            if self.piece_check(p2_copy, p2.copy()):
                print("--^^--")
                np = self.Tolerance(p1.copy(), p2.copy())
                if np != False:
                    if self.calc_check(p1.copy(), p2.copy(), np.copy()):
                        print("**^^**")
                        return np
                    return False
            else:
                print("--「変形」--")
                return False

    def calc_check(self, p1, p2, np):   #新しいピースの面積を調べる
        pp_area = self.calc_area(p1) + self.calc_area(p2)
        np_area = self.calc_area(np)

        if pp_area <= np_area:
            if pp_area * 1.3 >= np_area:    #多角形近似をした際に誤差が生じる可能性がある
                return True
        else:
            return False

    def reversePiece(self, piece , flag, refV):  #頂点refVを基準にして反転
        x = refV[0]
        y = refV[1]

        if flag == "reverse":
            for i in range(len(piece)):
                dx = x - piece[i][0]
                piece[i][0] = x + dx
        return piece

    #スコアが高い順にリストを生成
    def piece_bind(self, array, result):
        score_list = []
        totalScore = 0
        count = 0
        sfList = []
        for i in range(len(result)):
            comb, score = self.search_func(array[result[i][0]], array[result[i][1]])
            totalScore += score
            count += 1
            token = [comb, score]
            sfList.append(token)

        Threshold = float(totalScore / count)   #閾値
        print("閾値 = ", Threshold)
        for i in range(len(result)):
            piece = [array[result[i][0]], array[result[i][1]]]
            # comb, score = self.search_func(array[result[i][0]], array[result[i][1]])
            comb = sfList[i][0]
            score = sfList[i][1]
            if score >= Threshold:
                combination = [result[i][0], result[i][1]]
                list = [score, piece, comb, combination]
                score_list.append(list)  #score_list=[スコア, [p1, p2], [p1の頂点番号, p2の頂点番号], ピースの組み合わせ]

        # sorted_score_list = sorted(score_list, reverse=True)
        # print("test_sortedScoreList↓")
        # pp.pprint(sorted_score_list, width=50)
        print("ペア数", len(score_list))
        print(score_list)
        return len(score_list), score_list

    #ピースの組み合わせを生成
    def piece_catch(self, area_num):
        result = []
        flag = 0
        while flag!= 15:
            flag += 1
            rand1 = random.randrange(len(area_num) - 1)
            rand2 = random.randrange(len(area_num) - 1)
            tmp = area_num[rand1]
            area_num[rand1] = area_num[rand2]
            area_num[rand2] = tmp

        for pair in itertools.combinations(area_num, 2):
            judge = True
            # print(pair)
            if list(pair) not in self.False_list:
                for i in range(len(result)):
                    if pair[0] not in result[i] and pair[1] not in result[i]:
                        pass
                    else:
                        judge = False
            else:
                judge = False

            if judge:
                result.append(list(pair))

        # print("test_result", result)
        return result

    def degree_liner_set(self, p1):  # 基準ピースに対して線分を反転させる必要がある．
        list1 = []
        for i in range(len(p1)):
            if i == 0:
                degree = self.calc_Degree3(p1[len(p1) - 1], p1[i], p1[i + 1], p1)
                liner1 = self.abs_liner(p1[len(p1) - 1], p1[i])
                liner2 = self.abs_liner(p1[i + 1], p1[i])
            elif i == len(p1) - 1:
                degree = self.calc_Degree3(p1[i - 1], p1[i], p1[0], p1)
                liner1 = self.abs_liner(p1[i - 1], p1[i])
                liner2 = self.abs_liner(p1[0], p1[i])
            else:
                degree = self.calc_Degree3(p1[i - 1], p1[i], p1[i + 1], p1)
                liner1 = self.abs_liner(p1[i - 1], p1[i])
                liner2 = self.abs_liner(p1[i + 1], p1[i])
            list1.append([degree, liner1, liner2, p1[i]])
        return list1

    # 面積計算
    def calc_area(self, p1):
        area = 0
        if p1 == False:
            return area
        # print("p1", p1)
        for i in range(len(p1)):
            if i == (len(p1) - 1):
                area += (p1[i][0] - p1[0][0]) * (p1[i][1] + p1[0][1])
            else:
                area += (p1[i][0] - p1[i + 1][0]) * (p1[i][1] + p1[i + 1][1])
        return 0.5 * abs(area)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def area_total(self, arr):  # 面積順にする
        area_list = []
        area_num = []
        for i in range(len(arr)):
            area_list.append([i, self.calc_area(arr[i])])
        area_list = sorted(area_list, reverse=True, key=lambda x: x[1])
        for i in range(len(area_list)):
            area_num.append(area_list[i][0])
        return area_num

    def calc_totalArea(self, arr):  #ピースの面積を計算
        totalArea = 0
        for i in range(len(arr)):
            totalArea += self.calc_area(arr[i])
        return totalArea

# 辺同士がなす角を計算
    def calc_Degree3(self, x, y, z, poly):
        a = x.copy()
        b = y.copy()
        c = z.copy()

        v_a = [a[0] - b[0], a[1] - b[1]]
        v_c = [c[0] - b[0], c[1] - b[1]]
        length_vec_a = np.linalg.norm(v_a)
        length_vec_c = np.linalg.norm(v_c)
        inner_product = np.inner(v_a, v_c)
        degree = np.rad2deg(np.arccos(inner_product / (length_vec_a * length_vec_c)))
        vec = [b[0] + (v_a[0] + v_c[0]) * 0.1, b[1] + (v_a[1] + v_c[1]) * 0.1]
        # print(poly)
        # print(f"{b=} {vec=} {degree=}{self.wn(poly,vec)=}")
        if not self.wn(poly, vec):
            return degree
        else:
            return 360 - degree
        # print(vec)
        """if((c[0]* a[1]) - (a[0] * c[1])) < 0:
            return 360 - degree
        else:
            return degree"""
        if v_c[0] >= 0:
            return degree
        else:
            return 360 - degree

    # 2点同士がなす角
    def calc_Degree2(self, a, b):
        radian = math.atan2(b[1] - a[1], b[0] - a[0])
        return radian * 180 / math.pi

    # 回転
    def rotate_cie(self, coordinate, rotate, a): #(piece, 角度, 頂点)
        # print("どうぞよろしくお願いいたします。0",a,coordinate)
        cie = [[0] * 2 for i in range(len(coordinate))]
        # print(rotate)
        # print("なんじゃもんじゃおばけ------------------------------------------------------------",coordinate)
        for i in range(len(coordinate)):
            if coordinate[i] == a:
                tmp = i
        new = [0, 0]
        new[0] = a[0] * math.cos(math.radians(rotate)) - a[1] * math.sin(math.radians(rotate))
        new[1] = a[0] * math.sin(math.radians(rotate)) + a[1] * math.cos(math.radians(rotate))
        for i in range(len(coordinate)):
            cie[i][0] = round(
                coordinate[i][0] * math.cos(math.radians(rotate)) - coordinate[i][1] * math.sin(math.radians(rotate)))
            cie[i][1] = round(
                coordinate[i][0] * math.sin(math.radians(rotate)) + coordinate[i][1] * math.cos(math.radians(rotate)))
        # print("------------------------------------------------------------なんじゃもんじゃおばけ",cie)
        if a[0] < cie[tmp][0]:
            c_x = cie[tmp][0] - a[0]
            c_y = cie[tmp][1] - a[1]
            for i in range(len(cie)):
                cie[i][0] -= c_x
                cie[i][1] -= c_y
        else:
            c_x = a[0] - cie[tmp][0]
            c_y = a[1] - cie[tmp][1]
            for i in range(len(cie)):
                cie[i][0] += c_x
                cie[i][1] += c_y
        tmp = cie[0]
        tmp_num = 0
        new_list = []
        for i in range(1, len(cie)):
            tmp1 = cie[i]
            if tmp1[1] < tmp[1]:
                tmp = tmp1
                tmp_num = i
            elif tmp1[1] == tmp[1]:
                if tmp[0] > tmp1[0]:
                    tmp = tmp1
                    tmp_num = i
        for i in range(tmp_num, len(cie)):
            new_list.append(cie[i])
        for i in range(0, tmp_num):
            new_list.append(cie[i])

        # print(a)
        # print("cie",cie)
        return new_list

    def Tolerance(self, poly1, poly2):  #多角形近似
        over_v = []  # くっついている点
        over_num = []   # くっついている点の配列での順番を残している
        add_list = []   # poly1(多角形)に対して追加する点の配列
        delete_list = []    # poly1(多角形)に対して削除する点の配列
        flag = False
        flag2 = False
        delete = False
        # print(poly1,poly2)
        for i in range(len(poly1)):
            for j in range(len(poly2)):
                if poly1[i] == poly2[j]:    # 同じ座標であれば，くっついている場所(座標と順番)を保存
                    over_v.append(poly2[j])
                    over_num.append([i, j])
                    # print(poly1[i])
                    if i >= 1 and i < (len(poly1) - 1):  # 1つ目
                        ang_tmp1 = self.calc_Degree3(poly1[i - 1], poly1[i], poly1[i + 1], poly1)
                    elif i == 0:
                        ang_tmp1 = self.calc_Degree3(poly1[len(poly1) - 1], poly1[i], poly1[i + 1], poly1)
                    elif i == (len(poly1) - 1):
                        ang_tmp1 = self.calc_Degree3(poly1[i - 1], poly1[i], poly1[0], poly1)
                    if j >= 1 and j < (len(poly2) - 1):  # 2つ目
                        ang_tmp2 = self.calc_Degree3(poly2[j - 1], poly2[j], poly2[j + 1], poly2)
                    elif j == 0:
                        ang_tmp2 = self.calc_Degree3(poly2[len(poly2) - 1], poly2[j], poly2[j + 1], poly2)
                    elif j == (len(poly2) - 1):
                        ang_tmp2 = self.calc_Degree3(poly2[j - 1], poly2[j], poly2[0], poly2)
                    # print(ang_tmp1,ang_tmp2)
                    if ang_tmp1 + ang_tmp2 >= 345:  # 角度によって消滅
                        delete_list.append(poly1[i])
                        del over_v[-1]
                        del over_num[-1]  # TODO:poly1の方の頂点が削除できてない
                        flag = True
        if len(delete_list) > 0:
            for i in reversed(range(len(poly1))):
                for j in reversed(range(len(delete_list))):
                    if poly1[i] == delete_list[j]:
                        del poly1[i]
                        break
        # print("はやくおえおう",over_num,over_v)
        if len(over_v) > 1:
            # print(poly1)
            # print(poly2)
            for i in range(over_num[1][1] + 1, len(poly2)):  # 反時計回りで頂点を取得
                if poly2[i] != delete and poly2[i] not in poly1 and poly2[i] not in delete_list:
                    add_list.append(poly2[i])
            for i in range(0, over_num[0][1]):
                if poly2[i] != delete and poly2[i] not in poly1 and poly2[i] not in delete_list:
                    add_list.append(poly2[i])
            if add_list == []:
                flag2 = True
                for i in range(over_num[0][1], over_num[1][1]):
                    if poly2[i] != delete and poly2[i] not in poly1 and poly2[i] not in delete_list:
                        add_list.append(poly2[i])
            if len(over_v) < 1:
                for i in range(len(poly1) - 1):
                    for j in range(len(poly2)):
                        if self.linehit(poly1[i], poly1[i + 1], poly2[j]) or self.linehit(poly1[i + 1], poly1[i],
                                                                                          poly2[j]):  # 線上に追加されているかチェック
                            if (poly2[j] in add_list) == False:  # addlistになければ追加
                                for k in range(len(over_num)):
                                    if over_num[k][1] > j:
                                        add_list.insert(k, poly2[j])
                                        break
                                    elif over_num[len(over_num) - 1][1] < j:
                                        add_list.insert((len(over_num)), poly2[j])
                                        break
                                over_v.append(poly2[j])  # over_v　線上のやつ
            # print(f"{add_list=}")
            for j in range(len(add_list)):
                flag_add = True
                # print(over_num)
                """if len(add_list) == 1  and flag_add:
                    j += 1
                    flag_add = False"""
                if len(over_v) > 1:
                    if flag2:  # これいらないかもけしてもいい
                        if (add_list[j] in poly1) == False:
                            poly1.insert(over_num[0][0] + j + 1, add_list[j])  # もしかしたらミスってるかも
                    else:
                        if (add_list[j] in poly1) == False:
                            poly1.insert(over_num[1][0] + j, add_list[j])
                elif len(over_v) == 1:
                    # if flag2:
                    if (add_list[j] in poly1) == False:
                        poly1.insert(over_num[0][0] + j + 1, add_list[j])  # もしかしたらミスってるかも
        flag_stop = True
        while (flag_stop):
            for i in range(len(poly1)):
                if len(poly1) < 3:  #統合結果が多角形でないとき
                    print("頂点の数が３つ未満")
                    return False
                if i == 0:
                    # print("よくエラーが生じるとこ", poly1)
                    if (poly1[len(poly1) - 1][0] == poly1[i][0] and poly1[i + 1][0] == poly1[i][0]) or (
                            poly1[len(poly1) - 1][1] == poly1[i][1] and poly1[i + 1][1] == poly1[i][1]):
                        del poly1[i]
                        flag_stop == True
                        break
                elif i == len(poly1) - 1:
                    if (poly1[i - 1][0] == poly1[i][0] and poly1[0][0] == poly1[i][0]) or (
                            poly1[i - 1][1] == poly1[i][1] and poly1[0][1] == poly1[i][1]):
                        del poly1[i]
                        flag_stop == True
                        break
                else:
                    if (poly1[i - 1][0] == poly1[i][0] and poly1[i + 1][0] == poly1[i][0]) or (
                            poly1[i - 1][1] == poly1[i][1] and poly1[i + 1][1] == poly1[i][1]):
                        del poly1[i]
                        flag_stop == True
                        break
            flag_stop = False
        # print("Tolerance !!!!!!!!!!!!!!!",poly1)
        return poly1

    def route(self, cie, st, ter):  # 頂点組み換え
        new_list = []
        for i in range(len(cie)):
            if cie[i] == st[0]:
                start = i
            if cie[i] == ter[0]:
                end = i
        if start > end:
            tmp = start
            start = end
            end = tmp
        leng = abs(start - end)
        leng2 = abs(len(cie) - leng)
        if leng <= leng2:
            for i in (range(end, len(cie))):
                new_list.append(cie[i])
            for i in (range(0, start + 1)):
                new_list.append(cie[i])
        else:
            for i in range(start, end):
                new_list.append(cie[i])
        return new_list

    def wn(self, polygon, point):
        from shapely.geometry import Polygon, Point
        return not Point(point).within(Polygon(polygon))

    def linehit(self, a, b, p):
        if (a[0] <= p[0] and p[0] <= b[0]) or (b[0] <= p[0] and p[0] <= a[0]):
            if (a[1] <= p[1] and p[1] <= b[1]) or (b[1] <= p[1] and p[1] <= a[1]):
                if ((p[1] * (a[0] - b[0])) + (a[1] * (b[0] - p[0])) + (b[1] * (p[0] - a[0])) == 0):
                    return True
        return False

    def overlap(self, p1, p2, p3, p4):
        if p1[0] >= p2[0]:
            return False
        else:
            if (p2[0] < p3[0] and p2[0] < p4[0]) or (p1[0] > p3[0] and p1[0] > p4[0]):
                return False
        if p1[1] >= p2[1]:
            if (p1[1] < p3[1] and p1[1] < p4[1]) or (p2[1] > p3[1] and p2[1] > p4[1]):
                return False
        else:
            if (p2[1] < p3[1] and p2[1] < p4[1]) or (p1[1] > p3[1] and p1[1] > p4[1]):
                return False
        if ((p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])) * (p1[0] - p2[0]) * (
                p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0]) > 0:
            return False
        if ((p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])) * (p3[0] - p4[0]) * (
                p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0]) > 0:
            return False
        return True

    def allPiece_check(self, preRow, poly_row): #移動前後のピースの面積が変化していないか調べる
        flag = True
        count = 1
        for i in range(len(preRow)):
            if self.piece_check(preRow[i], poly_row[i]) != True:
                flag = False
                print('移動失敗' +str(count) + '\n移動前:' + str(preRow[i]) + '\n移動後:' + str(poly_row[i]))
                poly_row[i] = preRow[i]
                count += 1
        return flag, poly_row

if __name__ == "__main__":
    pa = paking_program()
    size = 8.0
    fig, ax = plt.subplots(figsize=(size, size))
    append_list = []
    disappear_list = []
    with open(pa.MyPath, encoding="utf-8_sig") as f:
        reader = csv.reader(f)
        for row in reader:
            row = [int(s) for s in row]
            pa.rows.append(row)
    # ピース配置
    for i in range(len(pa.rows)):
        q = int(len(pa.rows[i]) / 2)
        tmp = [[0] * 2 for i in [1] * q]
        k = 0
        for j in range(0, len(pa.rows[i]), 2):
            tmp[k][0] = pa.rows[i][j]
            tmp[k][1] = pa.rows[i][j + 1]
            k += 1
        pa.arr_piece.append(tmp)
    print("\n*arr_piece*")
    print("\n".join(map(str, pa.arr_piece)))
    pa.piece_data_backup = pa.arr_piece #移動前のピースデータを保存
    for i in range(len(pa.arr_piece)):
        tmp = [[0] * 2 for i in [1] * len(pa.arr_piece[i])]
        rand1 = random.randint(-15, 15)
        rand2 = random.randint(-15, 15)
        for k in range(0, len(pa.arr_piece[i])):
            tmp[k][0] = pa.arr_piece[i][k][0] + rand1
            tmp[k][1] = pa.arr_piece[i][k][1] + rand2
        pa.poly_row.append(tmp)
    checkFlag, pa.poly_row = pa.allPiece_check(pa.piece_data_backup, pa.poly_row)
    print("checkFlag :", checkFlag)
    print("\n*poly_row*")
    print("\n".join(map(str, pa.poly_row)))
    print("\n")
    nagasa = 0
    p_num = len(pa.poly_row)
    p_count = 0 #ピースが統合した数
    pa.piece_data_backup = []
    checkFlag = True
    # 統合
    loop = 10   #ループの回数
    while nagasa != loop:  # ループ回数管理
        nagasa += 1
        totalArea = pa.calc_totalArea(pa.poly_row)
        # print("totalArea = ", totalArea)
        # r = (totalArea ** (1/2)) * 0.2
        r = 3
        collision_list = pa.collision_rule(pa.poly_row, r)  # 円内検知
        area_num = pa.area_total(pa.poly_row)  # collision_list
        result = pa.piece_catch(area_num)
        if result == []:
            break
        print(f"{nagasa} {result}")
        # print("poly_row", pa.poly_row)
        tmp, comb = pa.piece_bind(pa.poly_row, result)  # collision_list
        # print("test", comb[0])
        # print("test", comb[0][1][0])
        # print(f"{tmp} {comb}")
        exclusion_list = []

        add_list = []
        new_list = []
        flag = True
        print("最終的なペア数", tmp)
        # print("comb", comb)
        if tmp != False:
            if tmp == 1:
                new_piece = pa.piece_move(comb[0][1], comb[0][2])
                if new_piece == False:
                    # pa.False_list.append(comb[0][3]) #いるかわからん
                    # print("False_list", pa.False_list)
                    # print("fail1")
                    flag = False
                else:
                    print("piece1,piece2 = ", comb[0][1][0], comb[0][1][1])
                    print("new_piece", new_piece)
                    exclusion_list.append(comb[0][1][0])
                    exclusion_list.append(comb[0][1][1])
                    add_list.append(new_piece)
                    append_list.append(new_piece)
                    # print("sucess1")
                    flag = True
            else:
                for i in range(tmp-1):  # poly_rowはcollision_listに書き換える
                    new_piece = pa.piece_move(comb[i][1], comb[i][2])
                    if new_piece == False:
                        # pa.False_list.append(comb[i][3]) # いるかわからん
                        # print("False_list", pa.False_list)
                        # print("fail2")
                        flag = False
                    else:
                        print("new_piece", new_piece)
                        print("comb", comb[i][3][0], comb[i][3][1])
                        exclusion_list.append(comb[i][1][0])
                        exclusion_list.append(comb[i][1][1])
                        add_list.append(new_piece)
                        append_list.append(new_piece)
                        # print("sucess2")
                        flag = True
            if flag:
                exclusion_list = sorted(exclusion_list, reverse=True)
                print("exclusion_list", exclusion_list)
                # print("poly_row", pa.poly_row)
                for i in range(len(exclusion_list)):
                    for j in range(len(pa.poly_row)):
                        if pa.poly_row[j] not in exclusion_list:
                            if pa.poly_row[j] not in new_list:
                                new_list.append(pa.poly_row[j])
                for i in range(len(add_list)):
                    if add_list[i] not in new_list:
                        p_count += 1
                        new_list.append(add_list[i])
                print("addlist", add_list)
                print("new_list", new_list)
                pa.poly_row = new_list
                # print("poly_row", pa.poly_row)
                pa.False_list = []
                disappear_list.append(exclusion_list)
                exclusion_list = []
        print("False_list", pa.False_list)
        print("<poly_row>", pa.poly_row)

        pa.piece_data_backup = pa.poly_row
        for i in range(len(pa.poly_row)):
            randx = random.randint(-8, 8)
            randy = random.randint(-8, 8)
            dx = randx - pa.poly_row[i][0][0]
            dy = randy - pa.poly_row[i][0][1]
            for j in range(len(pa.poly_row[i])):
                pa.poly_row[i][j][0] += dx
                pa.poly_row[i][j][1] += dy
        checkFlag, pa.poly_row = pa.allPiece_check(pa.piece_data_backup, pa.poly_row)
        print("checkFlag :", checkFlag)
        print("\n**poly_row**")
        print("\n".join(map(str, pa.poly_row)))
        print("\n")
        #ピースが1つに統合されたら処理終了
        if len(pa.poly_row) == 1:
            nagasa = loop

    pa.piece_data_backup = pa.poly_row
    for i in range(len(pa.poly_row)):
        rand1 = random.randint(-20, 20)
        rand2 = random.randint(-20, 20)
        for k in range(0, len(pa.poly_row[i])):
            pa.poly_row[i][k][0] = pa.poly_row[i][k][0] + rand1
            pa.poly_row[i][k][1] = pa.poly_row[i][k][1] + rand2
        ax.add_patch(patches.Polygon(pa.poly_row[i], color=pa.colors[5], fill=False, linewidth=2))
    checkFlag, pa.poly_row = pa.allPiece_check(pa.piece_data_backup, pa.poly_row)
    print("checkFlag :", checkFlag)
    print("---poly_row---")
    print("\n".join(map(str, pa.poly_row)))
    print("\n")

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False,
                    right=False, top=False)

    ax.autoscale()
    print("append_list:", append_list)
    print("disappear_list:", disappear_list)
    print("結合した数, ピース数 : ", p_count, len(pa.poly_row))
    plt.show()
