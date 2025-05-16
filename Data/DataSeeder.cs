using Bogus;
using BurnAnalysisApp.Models; 
using BurnAnalysisApp.Data;
using System.Linq;

public class DataSeeder
{
    public static void Seed(IServiceProvider serviceProvider)
    {
        var context = serviceProvider.GetRequiredService<AppDbContext>();

        // Admin verisi ekleme
        if (!context.Admins.Any())
        {
            var adminFaker = new Faker<Admin>()
                .RuleFor(a => a.Name, f => f.Name.FullName())
                .RuleFor(a => a.Email, f => f.Internet.Email())
                .RuleFor(a => a.Password, f => f.Internet.Password());

            var admins = adminFaker.Generate(10);
            context.Admins.AddRange(admins);
        }

        // Doktor verisi ekleme
        if (!context.Doctors.Any())
        {
            var doctorFaker = new Faker<Doctor>()
                .RuleFor(d => d.Name, f => f.Random.Bool() ? f.Name.FullName() : null)
                .RuleFor(d => d.Email, f => f.Internet.Email())
                .RuleFor(d => d.Password, f => f.Internet.Password())
                .RuleFor(d => d.Verified, f => f.Random.Bool())
                .RuleFor(d => d.AdminID, f => f.PickRandom(context.Admins.Select(a => a.AdminID).ToList()));

            var doctors = doctorFaker.Generate(10);
            context.Doctors.AddRange(doctors);
        }

        // Hasta verisi ekleme
        if (!context.Patients.Any())
        {
            var patientFaker = new Faker<PatientInfo>()
                .RuleFor(p => p.Name, f => f.Name.FullName())
                .RuleFor(p => p.Age, f => f.Random.Int(0, 100)) // Yaşı rastgele 0-100 arasında
                .RuleFor(p => p.Gender, f => f.PickRandom(new[] { "Male", "Female", "Other" }))
                .RuleFor(p => p.MedicalHistory, f => f.Lorem.Sentence(5)); // Kısa bir geçmiş

            var patients = patientFaker.Generate(5); // 5 hasta oluştur
            context.Patients.AddRange(patients);
        }

        // BurnImage verisi ekleme
        if (!context.BurnImages.Any())
        {
            var burnImageFaker = new Faker<BurnImage>()
                .RuleFor(b => b.PatientID, f => f.PickRandom(context.Patients.Select(p => p.PatientID).ToList()))
                .RuleFor(b => b.UploadedBy, f => f.PickRandom(context.Doctors.Select(d => d.DoctorID).ToList()))
                .RuleFor(b => b.FilePath, f => f.System.FilePath()) // Rastgele dosya yolu
                .RuleFor(b => b.UploadDate, f => f.Date.Past(1)) // Son bir yıl içinde bir tarih
                .RuleFor(b => b.BurnDepth, f => f.PickRandom(new[] { "Superficial", "Partial Thickness", "Full Thickness" }))
                .RuleFor(b => b.SurfaceArea, f => f.Random.Float(1, 99)) // Yüzde 1-99 arası
                .RuleFor(b => b.BodyLocation, f => f.PickRandom(new[] { "Arm", "Leg", "Torso", "Head", "Hand" }))
                .RuleFor(b => b.SeverityScore, f => f.Random.Float(1, 10)) // 1-10 arasında bir ciddiyet puanı
                .RuleFor(b => b.AnalysisDate, f => f.Date.Past(1)) // Analiz tarihi
                .RuleFor(b => b.SharedWithDoctors, f => f.Random.Bool());

            var burnImages = burnImageFaker.Generate(5); // 5 burn image kaydı oluştur
            context.BurnImages.AddRange(burnImages);
        }

        // Verileri kaydet
        context.SaveChanges();
    }
}
