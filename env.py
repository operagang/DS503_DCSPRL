import copy, sys

class Simulator:
    def __init__(self, inst):
        self.inst = inst
        self.crane_lag = self.inst['crane_lag_dict']
        self.unit_time = self.inst['unit_time']
        self.handling_time = self.inst['handling_time']
        self.reset()
        self.before = 0

    def reset(self):
        self.arrival_dict = copy.deepcopy(self.inst['arrival_dict'])
        self.arrival_list = list(self.arrival_dict.keys())
        self.clock = min(self.arrival_list)
        self.arrival_list.remove(self.clock)

        self.task_dict = self.set_task()
        self.task_list = list(self.task_dict.keys())
        self.crane_dict = self.set_crane()
        self.crane_list = list(self.crane_dict.keys())
        
        self.assigned_tasks = []
        self.unassigned_task = copy.copy(self.task_list)
        
        self.cur_task = copy.copy(self.arrival_dict[self.clock])
        self.sched = self.Schedule()
        self.check_wait = {self.clock:{i:False for i in self.crane_list}}
        self.trajectory = {i:[self.crane_dict[i].pos] for i in self.crane_list}

    class Task:
        def __init__(self, s, t, dur, arr):
            self.s = s
            self.t = t
            self.dur = dur
            self.arr = arr
            self.last_crane = None
    
    class Crane:
        def __init__(self, pos, left, right, sim):
            self.pos = pos
            self.assigned = False
            self.started = False
            self.s = None
            self.t = None
            self.avail_time = sim.clock
            self.task = None
            self.left = left
            self.right = right

    class Schedule:
        def __init__(self):
            self.start_time = {}
            self.compl_time = {}
            self.task_crane = {}

    def set_task(self):
        task_dict = {}
        for i,dict_ in self.inst['task_dict'].items():
            from_ = dict_['from']
            to_ = dict_['to']
            duration = abs(from_-to_)*self.inst['unit_time'] + self.inst['handling_time']*2
            arrival = dict_['arrival']
            task = self.Task(from_, to_, duration, arrival)
            task_dict[i] = task
        return task_dict
    
    def set_crane(self):
        crane_dict = {}
        for i,dict_ in self.inst['crane_dict'].items():
            pos = dict_['init_pos']
            if (i != 0):
                left = i-1
            else:
                left = None
            if (i != len(self.inst['crane_dict'])-1):
                right = i+1
            else:
                right = None
            crane = self.Crane(pos, left, right, self)
            crane_dict[i] = crane
        return crane_dict
    
    def update(self, task, crane):
        task_obj = self.task_dict[task]
        crane_obj = self.crane_dict[crane]
        duration = task_obj.dur

        # crane 이동
        crane_pos = crane_obj.pos
        task_pos = task_obj.s
        start = self.clock + abs(crane_pos - task_pos)*self.unit_time
        
        # 간섭
        for prev_task in self.assigned_tasks:
            prev_crane = self.task_dict[prev_task].last_crane
            prev_compl = self.sched.compl_time[prev_task]
            start = max(start, prev_compl + self.crane_lag[prev_task, task, prev_crane, crane])
        
        # task state update
        task_obj.last_crane = crane

        # crane state update
        crane_obj.assigned = True
        crane_obj.s = task_obj.s
        crane_obj.t = task_obj.t
        crane_obj.avail_time = start + duration
        crane_obj.task = task
        if self.clock == start:
            crane_obj.started = True

        # schedule update
        self.sched.start_time[task] = start
        self.sched.compl_time[task] = start + duration
        self.sched.task_crane[task] = crane

        # 할당정보 update
        self.assigned_tasks.append(task)
        self.unassigned_task.remove(task)
        self.cur_task.remove(task)

    def get_next_clock(self):
        next_clock = float('inf')
        
        event_list = []
        for crane in self.crane_list:
            crane_obj = self.crane_dict[crane]
            if crane_obj.assigned:
                event_list.append((crane_obj.avail_time, 'crane'))
            elif not self.check_wait[self.clock][crane]:
                event_list.append((self.clock, 'crane'))
            elif len(self.arrival_list) > 0:
                event_list.append((min(self.arrival_list), 'crane'))
        for arr in self.arrival_list:
            event_list.append((arr, 'task'))
        if len(self.cur_task) > 0:
            event_list.append((self.clock, 'task'))

        event_list.sort(key=lambda x:x[0])
        task_avail = False
        crane_avail = False
        for event in event_list:
            if event[1] == 'task':
                task_avail = True
            elif event[1] == 'crane':
                crane_avail = True
            if task_avail and crane_avail:
                next_clock = event[0]
                break

        # # event 1: 작업할 task 있는경우 -> 가장빠른 crane 작업 종료
        # if len(self.cur_task):
        #     for crane in self.crane_list:
        #         crane_obj = self.crane_dict[crane]
        #         if crane_obj.assigned:
        #             next_clock = min(next_clock, crane_obj.avail_time)
        #         elif not self.check_wait[self.clock][crane]:
        #             # 작업할 task 있고, 현재 wait 선택 안한 available crane 존재 시
        #             return self.clock
        
        # # event 2: available crane 존재 -> 가장 빠른 arrival
        # avail_count = 0
        # for crane in self.crane_list:
        #     if not self.crane_dict[crane].assigned:
        #         avail_count += 1
        # if avail_count > 0:
        #     if len(self.arrival_list):
        #         next_clock = min(next_clock, min(self.arrival_list))
        
        # # event 3: 현재는 작업할 task 없고, arrival 후 가장 빠른 available 시점
        # c = 0
        # if len(self.cur_task)==0 and avail_count==0 and len(self.arrival_list)>0:
        #     self.arrival_list.sort()
        #     for arr in self.arrival_list:
        #         for crane in self.crane_list:
        #             crane_obj = self.crane_dict[crane]
        #             if crane_obj.avail_time <= arr:
        #                 next_clock = min(next_clock, arr)
        #                 c += 1
        #                 break
        #         if c > 0:
        #             break

        # # 에러상황
        # if (len(self.arrival_list) == 0) and (avail_count == len(self.crane_list)):
        #     sys.exit('New arrival 없고, 종료될 crane도 없음. Next event 없음')
        if next_clock == float('inf'):
            sys.exit('왠진모르겠지만 다음 시점이 inf임')

        return next_clock

    def move_to_next_clock(self, next_clock):
        for clock in range(self.clock+1, next_clock+1):
            # new arrival
            if len(self.arrival_list) > 0:
                if clock == min(self.arrival_list):
                    self.arrival_list.remove(clock)
                    for task in self.arrival_dict[clock]:
                        self.cur_task.append(task)

            # crane 위치 수정
            moved = {crane:False for crane in self.crane_list}
            for crane in self.crane_list:
                crane_obj = self.crane_dict[crane]

                ## 시작한 크레인
                if crane_obj.started:
                    task_obj = self.task_dict[crane_obj.task]
                    start = self.sched.start_time[crane_obj.task]
                    if clock <= start + self.handling_time:
                        crane_obj.pos = crane_obj.pos
                        moved[crane] = True
                        if crane_obj.pos != task_obj.s:
                            sys.exit('trajectory error1')
                    elif clock <= start + (task_obj.dur - self.handling_time):
                        if crane_obj.pos < task_obj.t:
                            crane_obj.pos += 1
                            moved[crane] = True
                        elif crane_obj.pos > task_obj.t:
                            crane_obj.pos -= 1
                            moved[crane] = True
                        else:
                            sys.exit('trajectory error2')
                    else:
                        crane_obj.pos = crane_obj.pos
                        moved[crane] = True
                        if crane_obj.pos != task_obj.t:
                            sys.exit('trajectory error3')
                ## assign된 크레인
                elif crane_obj.assigned:
                    task_obj = self.task_dict[crane_obj.task]
                    start = self.sched.start_time[crane_obj.task]
                    min_travel = abs(crane_obj.pos - task_obj.s)*self.unit_time
                    if (start - clock) < min_travel:
                        if crane_obj.pos < task_obj.s:
                            crane_obj.pos += 1
                            moved[crane] = True
                        elif crane_obj.pos > task_obj.s:
                            crane_obj.pos -= 1
                            moved[crane] = True
                        else:
                            sys.exit('trajectory error4')
            
            ## 잉여 크레인 옮기기
            for crane in self.crane_list:
                crane_obj = self.crane_dict[crane]
                if moved[crane]:
                    left_crane = crane_obj.left
                    while left_crane != None:
                        left_obj = self.crane_dict[left_crane]
                        if left_obj.pos == self.crane_dict[left_obj.right].pos:
                            left_obj.pos -= 1
                            left_crane = left_obj.left
                        else:
                            break
                    right_crane = crane_obj.right
                    while right_crane != None:
                        right_obj = self.crane_dict[right_crane]
                        if right_obj.pos == self.crane_dict[right_obj.left].pos:
                            right_obj.pos += 1
                            right_crane = right_obj.right
                        else:
                            break
                        
            for crane in self.crane_list:
                crane_obj = self.crane_dict[crane]
                self.trajectory[crane].append(crane_obj.pos)

                if crane_obj.left == None:
                    continue
                elif crane_obj.pos <= self.crane_dict[crane_obj.left].pos:
                    sys.exit('trajectory error5')

            # crane state update
            for crane in self.crane_list:
                crane_obj = self.crane_dict[crane]
                if crane_obj.assigned:
                    if not crane_obj.started:
                        start = self.sched.start_time[crane_obj.task]
                        if clock == start:
                            crane_obj.started = True
                    else:
                        compl = self.sched.compl_time[crane_obj.task]
                        if clock == compl:
                            crane_obj.assigned = False
                            crane_obj.started = False
                            crane_obj.s = None
                            crane_obj.t = None
                            crane_obj.task = None
        
        self.clock = next_clock
        self.check_wait[self.clock] = {i:False for i in self.crane_list}

    def terminate(self):
        obj_value = 0
        for task in self.task_list:
            obj_value += self.sched.compl_time[task]
        self.obj_value = obj_value
        # S_T로 변경은 나중에

    def step(self, action):
        task = action[0]
        crane = action[1]

        if task != None:
            self.update(task, crane)
        else:
            self.check_wait[self.clock][crane] = True

        # termination 확인
        if len(self.unassigned_task) == 0:
            self.terminate()
            done = True

            return self.get_state(), 0, done
            # return self.get_state(), -self.obj_value, done

        # 다음 시점 계산
        next_clock = self.get_next_clock()
        h1 = 0 
        for task in range(len(self.task_dict)):
            tmp = self.task_dict[task]
            if tmp.arr <= self.clock and task not in self.assigned_tasks:
                h1 += 1 
            if task in self.assigned_tasks:
                if self.before < self.task_dict[task].t:
                    h1 += 1
                
        reward = -(next_clock - self.before)*h1
        # print("reward", reward)
        self.before = copy.deepcopy(next_clock)
        
        # 다음 시점으로 이동
        if next_clock > self.clock:
            self.move_to_next_clock(next_clock)
        done = False
        return self.get_state(), reward, done
    
        # return self.get_state(), None, done

    def get_state(self):
        job_state = []
        # masking 용으로 변함
        for task in range(len(self.task_dict)):
            tmp = self.task_dict[task]
            if tmp.arr <= self.clock and task not in self.assigned_tasks:
                job_state.append([0, tmp.s, tmp.t, tmp.dur])
            else:
                job_state.append([1, tmp.s, tmp.t, tmp.dur])
                
        #  [할당여부, 시작여부, 현재 위치, from, to, 시간] 
        crane_state = []
        for crane in range(len(self.crane_dict)):
            tmp = self.crane_dict[crane]
            if tmp.assigned == False:
                crane_state.append([0, 0, tmp.pos, 0, 0, 0 ])
            else:
                if tmp.started:
                    crane_state.append([1, 1, tmp.pos, tmp.s, tmp.t, tmp.avail_time - self.clock] )
                else:
                    crane_state.append([1, 0, tmp.pos, tmp.s, tmp.t, tmp.avail_time - self.clock] )
        return job_state, crane_state