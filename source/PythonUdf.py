from . import *

class PythonUdf():

    '''
    contains examples of python udfs
    '''

    @F.udf("array<float>")
    def sparse_to_array(v):
      '''
      https://danvatterott.com/blog/2018/07/08/aggregating-sparse-and-dense-vectors-in-pyspark/
      converts a sparse vector to an array and returns it
      *v*: sparse vector
      '''
      v = DenseVector(v)
      new_array = list([float(x) for x in v])
      return new_array

    @F.udf("array<int>")
    def weeks_before_birthday(datestamp,
                              birthdate,
                              max_week =4):
        '''
        returns a list containing boolean flags of the form (x weeks from birthday) for x in (1,max_week)
        *datestamp*: the date of a certain payment interaction
        *birthdate*: the birthdate of the customer
        *max_week*: the number of binary columns to create corresponding to the number of weeks between an interaction and the customer's birthday
        '''
        if birthdate == None:
            return list(map(lambda w: int(w), np.zeros(max_week + 1)))
        cur_year_birthday_day = birthdate.day if (birthdate.day <= calendar.monthrange(datestamp.year, birthdate.month)[1]) else calendar.monthrange(datestamp.year, birthdate.month)[1]
        next_year_birthday_day = birthdate.day if (birthdate.day <= calendar.monthrange(datestamp.year + 1, birthdate.month)[1]) else calendar.monthrange(datestamp.year + 1, birthdate.month)[1]
        birthday = datetime.datetime(datestamp.year + 1, birthdate.month, next_year_birthday_day) if (datestamp > datetime.datetime(datestamp.year, birthdate.month, cur_year_birthday_day).date()) else datetime.datetime(datestamp.year, birthdate.month, cur_year_birthday_day)
        days_from_birthday = (birthday.date() - datestamp).days
        return list(map(lambda w: 1 if (math.floor(days_from_birthday / 7) == w) else 0, list(range(max_week + 1))))
