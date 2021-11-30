
import numpy as np

class simulator:

    # Room Generations Plans
    def generate_clumpy_plan(N,p,room, clump_size = 3):
        """ Generates a seating plan where the unvaccinated students sit together
        in clumps. Represents the worst case scenario.
        """
        Nvax = np.random.binomial(N,p)
        Nunvax = N-Nvax
        room = room.drop('UnvaxSpot',axis = 1).reset_index()
        grid = room.copy()
        grid['seating'] = 'E'

        if Nunvax > 0:
            clump_size = min(clump_size,Nunvax)
            num_clumps = math.floor(Nunvax/clump_size)
            remainder = Nunvax - num_clumps*clump_size
            clump = 0
            while clump < (num_clumps):
                ind1 = np.random.choice(room['index'].values,replace = False)
                grid['seating'].loc[grid['index'] == ind1] = 'U'
                room = room.drop(ind1, axis = 0)
                x_temp = grid['x'].loc[grid['index'] == ind1].values[0]
                y_temp = grid['y'].loc[grid['index'] == ind1].values[0]
                temp = room.copy()
                temp['dist_infected'] = ((temp['x'] -x_temp) ** 2 + (temp['y'] - y_temp) ** 2) ** 0.5
                temp = temp.sort_values('dist_infected', ascending = True).head(clump_size-1)
                grid['seating'].loc[grid['index'].isin(temp['index'].values)] = 'U'
                room = room.drop(temp['index'].values, axis = 0)
                clump = clump + 1
            remainder_ind = np.random.choice(grid['index'].loc[grid['seating'] == 'E'],remainder,replace = False)
            grid['seating'].loc[grid['index'].isin(remainder_ind)] = 'U'

        vax_ind = np.random.choice(grid['index'].loc[grid['seating'] == 'E'],Nvax,replace = False)
        grid['seating'].loc[grid['index'].isin(vax_ind)] = 'V'

        return grid

    def generate_random_plan(N,p,room):
        """ Generates a seating plan where students sit randomly in the room
        """
        Nvax = round(N*p)
        Nunvax = N-Nvax

        room = room.drop('UnvaxSpot',axis = 1)
        grid = room.copy()
        grid = grid.reset_index()
        temp = list(np.append(np.append(np.repeat('V',Nvax),np.repeat('U',Nunvax)),np.repeat('E',len(grid)-Nvax-Nunvax)))
        random.shuffle(temp)
        grid['seating'] = temp
        return grid
