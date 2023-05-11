from django.db import models
from django.contrib.auth.models import *

#custom User Manager
class UserManager(BaseUserManager):
    def create_user(self, email, firstname, lastname, password=None):
        
        if not email:
            raise ValueError('Users must have an email address')

        user = self.model(
            email=self.normalize_email(email),
            firstname=firstname,
            lastname=lastname,
                         )

        user.set_password(password)
        user.save(using=self._db)
        return user
    def create_superuser(self, email, password=None):
       
        user = self.create_user(
            email,
            firstname= "None",
            lastname= "None",
            password=password,
        )
        user.is_admin = True
        user.save(using=self._db)
        return user

#  Custom User Model
class User(AbstractBaseUser,PermissionsMixin):
    email = models.EmailField(
        verbose_name='email address',
        max_length=255,
        unique=True,
    )
    firstname = models.CharField(max_length=80)
    lastname = models.CharField(max_length=80)
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


    objects = UserManager()

    USERNAME_FIELD = 'email'

    REQUIRED_FIELDS = []


    def ___str__(self):
        return self.email

    def has_perm(self, perm, obj=None):
        "Does the user have a specific permission?"
        return self.is_admin


    def has_module_perms(self, app_label):
        "Does the user have permissions to view the app `app_label`?"
        return True

    @property
    def is_staff(self):
        "Is the user a member of staff?"
        return self.is_admin


LANGUAGE_CHOICES = (
    ('en', 'english'),('af','afrikaans'),('sq','albanian'),('am','amharic'),('ar','arabic'),
    ('hy', 'armenian'),('az','azerbaijani'),('eu','basque'),('be','belarusian'),('bn','bengali'),
    ('bs', 'bosnian'),('bg','bulgarian'),('ca','catalan'),('ceb','cebuano'),('ny','chichewa'),
    ('zh-cn', 'chinese (simplified)'),('zh-tw','chinese (traditional)'),('co','corsican'),
    ('hr','croatian'),('cs','czech'),('da','danish'),('nl','dutch'),('eo','esperanto'),
    ('et','estonian'),('tl','filipino'),('fi','finnish'),('fr','french'),('fy','frisian'),('gl','galician'),
    ('ka','georgian'),('de','german'),('el','greek'),('gu','gujarati'),('ht','haitian creole'),('ha','hausa'),
    ('haw','hawaiian'),('hi','hindi'),('hmn','hmong'),('hu','hungarian'),('is','icelandic'),
    ('ig','igbo'),('id','indonesian'),('ga','irish'),('it','italian'),('ja','japanese'),('jw','javanese'),
    ('kn','kannada'),('kk','kazakh'),('km','khmer'),('ko','korean'),('ku','kurdish (kurmanji)'),('ky','kyrgyz'),
    ('lo','lao'),('la','latin'),('lv','latvian'),('lt','lithuanian'),('lb','luxembourgish'),('mk','macedonian'),
    ('mg','malagasy'),('ms','malay'),('ml','malayalam'),('mt','maltese'),('mi','maori'),('mr','marathi'),
    ('mn','mongolian'),('my','myanmar (burmese)'),('ne','nepali'),('no','norwegian'),('ps','pashto'),('fa','persian'),
    ('pl','polish'),('pt','portuguese'),('pa','punjabi'),('ro','romanian'),('ru','russian'),('sm','samoan'),
    ('gd','scots gaelic'),('sr','serbian'),('st','sesotho'),('sn','shona'),('sd','sindhi'),('si','sinhala'),
    ('sk','slovak'),('sl','slovenian'),('so','somali'),('es','spanish'),('su','sundanese'),('sw','swahili'),
    ('sv','swedish'),('tg','tajik'),('ta','tamil'),('te','telugu'),('th','thai'),('tr','turkish'),
    ('uk','ukrainian'),('ur','urdu'),('uz','uzbek'),('vi','vietnamese'),('cy','welsh'),('xh','xhosa'),
    ('yi','yiddish'),('yo','yoruba'),('zu','zulu'),('he','Hebrew'),('fil','Filipino')
    ) 

VARIANT_CHOICES=(
    ("Cricket","Cricket"),("Mobile","Mobile"),("Technology","Technology")
)

class Content(models.Model):
    user=models.ForeignKey(User,on_delete=models.CASCADE)
    input=models.TextField()
    output=models.TextField(null=True,blank=True)
    variant=models.CharField(choices=VARIANT_CHOICES,max_length=30)
    language=models.CharField(choices=LANGUAGE_CHOICES,max_length=30)

class Cricket_Question_and_Answer(models.Model):
     question=models.TextField(max_length=1000)
     answer=models.TextField(max_length=1000)
    
    # class Meta:   
    #     app_label = 'firstmyapp'
    #     db_table ="cricket_question_and_answer"
    # using = 'default'

class Mobile_Technology_Waves(models.Model):
    question=models.TextField(max_length=1000)
    answer=models.TextField(max_length=1000)

    # class Meta:
    #     app_label = 'firstmyapp'
    #     db_table ="mobile_technology_waves"

    # # specify the database to use
    # using = 'default'
    
class Technologies(models.Model):
    question=models.TextField(max_length=1000)
    answer=models.TextField(max_length=1000)

    # class Meta:
    #     app_label = 'firstmyapp'
    #     db_table ="technologies"
    # using = 'user1'


class UserDataModels(models.Model):
    input=models.TextField()
    database_type=models.CharField(choices=VARIANT_CHOICES,max_length=30)
    label=models.TextField()